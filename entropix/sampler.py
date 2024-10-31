from typing import Dict, Tuple
import jax
import jax.numpy as jnp

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

@jax.jit
def calculate_varentropy_logsoftmax(logits: jnp.ndarray, axis: int = -1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax.
    
    As discussed in the thread, varentropy measures overall uncertainty/entropy variance.
    This implementation:
    1. Converts logits to probabilities via logsoftmax
    2. Calculates entropy (uncertainty per step)
    3. Calculates varentropy (variance in uncertainty)
    """
    log_probs = jax.nn.log_softmax(logits, axis=axis)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * log_probs, axis=axis) / LN_2  
    varentropy = jnp.sum(probs * (log_probs / LN_2 + entropy[..., None])**2, axis=axis)
    return entropy, varentropy

@jax.jit
def multinomial_sample_one(probs_sort: jax.Array, key) -> jax.Array:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = jax.random.exponential(key=key, shape=probs_sort.shape)
    return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)

@jax.jit
def _sample(logits: jax.Array, *, temperature: float | jax.Array, top_p: float | jax.Array, 
           top_k: int | jax.Array, min_p: float | jax.Array, key=jax.random.PRNGKey(1337)) -> jax.Array:
    """Basic sampling function with temperature, top-p, top-k, and min-p controls.
    
    This implements the core sampling strategy that corresponds to different quadrants
    in the entropy/varentropy framework shown in the image.
    """
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = jax.nn.softmax(logit / temperature, axis=-1)

    # Apply min-p sampling (for low entropy cases)
    if min_p > 0.0:
        p_max = jnp.max(probs, axis=-1, keepdims=True)
        indices_to_remove = probs < (min_p * p_max)
        logit = jnp.where(indices_to_remove, jnp.full_like(logit, float('-inf')), logit)

    # Apply top-k sampling (handles high varentropy cases)
    top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
    probs_sort = jnp.flip(top_k_probs, axis=-1)
    probs_idx = jnp.flip(top_k_indices, axis=-1)
    probs_sum = jnp.cumsum(probs_sort, axis=-1)
    
    # Apply top-p sampling (handles high entropy cases)
    mask = jnp.where(probs_sum - probs_sort > top_p, 1.0, 0.0)
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)
    
    next_token = multinomial_sample_one(probs_sort, key)
    next_token_g = jnp.take_along_axis(probs_idx, next_token.reshape(bsz, 1), axis=-1)
    return next_token_g.astype(jnp.int32)

@jax.jit
def adaptive_sample(logits: jax.Array, *, temperature: float | jax.Array, 
                   key=jax.random.PRNGKey(1337), epsilon: float = 0.01) -> jax.Array:
    """Adaptive sampling based on entropy changes.
    
    This implements what @_xjdr described as measuring "between logits per step"
    and adapting the sampling strategy based on entropy changes.
    """
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = jax.nn.softmax(logit / temperature, axis=-1)

    sorted_probs, sorted_indices = jax.lax.top_k(probs, k=probs.shape[-1])
    candidate_mask = jnp.zeros_like(sorted_probs, dtype=bool)
    cumulative_entropy = jnp.zeros((bsz,))
    cumulative_varentropy = jnp.zeros((bsz,))
    previous_entropy = -jnp.sum(sorted_probs[0] * jnp.log2(jnp.clip(sorted_probs[0], 1e-10, 1.0)))

    def cond_fn(state):
        cumulative_entropy, cumulative_varentropy, i, mask = state
        entropy_reduction = cumulative_entropy - previous_entropy
        return (entropy_reduction >= epsilon) & (i < sorted_probs.shape[-1])

    def body_fn(state):
        cumulative_entropy, cumulative_varentropy, i, mask = state
        current_prob = sorted_probs[:, i]
        
        # Calculate step-wise entropy and varentropy
        current_entropy = -jnp.sum(current_prob * jnp.log2(jnp.clip(current_prob, 1e-10, 1.0)))
        current_varentropy = jnp.sum(current_prob * (jnp.log2(jnp.clip(current_prob, 1e-10, 1.0)) + cumulative_entropy[:, None])**2)
        
        entropy_reduction = cumulative_entropy - current_entropy
        
        mask = jnp.where(entropy_reduction >= epsilon, mask.at[:, i].set(True), mask)
        
        cumulative_entropy = cumulative_entropy.at[:, i].set(current_entropy)
        cumulative_varentropy = cumulative_varentropy.at[:, i].set(current_varentropy)
        
        return cumulative_entropy, cumulative_varentropy, i + 1, mask

    initial_state = (cumulative_entropy, cumulative_varentropy, 0, candidate_mask)
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    
    final_mask = final_state[-1]
    candidate_probs = sorted_probs * final_mask
    candidate_probs = candidate_probs / jnp.sum(candidate_probs, axis=-1, keepdims=True)
    
    next_token = multinomial_sample_one(candidate_probs, key)
    next_token_g = jnp.take_along_axis(sorted_indices, next_token.reshape(bsz, 1), axis=-1)
    
    return next_token_g.astype(jnp.int32)

@jax.jit
def calculate_metrics(logits: jnp.ndarray, attention_scores: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Calculate various metrics from logits and attention scores.
    
    This implementation aligns with the quadrant framework shown in the image:
    - Measures both entropy (uncertainty) and varentropy (variance in uncertainty)
    - Tracks attention patterns which can indicate when to use different sampling strategies
    """
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = jax.nn.softmax(attention_scores, axis=-1)
    attn_entropy = -jnp.sum(attention_probs * jnp.log2(jnp.clip(attention_probs, 1e-10, 1.0)), axis=-1)
    attn_varentropy = jnp.var(attn_entropy, axis=1)

    mean_attention = jnp.mean(attention_probs, axis=1)
    agreement = jnp.mean(jnp.abs(attention_probs - mean_attention[:, None, :]), axis=(1, 2))
    interaction_strength = jnp.mean(jnp.abs(attention_scores), axis=(1, 2, 3))

    return {
        "logits_entropy": jnp.mean(entropy),
        "logits_varentropy": jnp.mean(varentropy),
        "attn_entropy": jnp.mean(attn_entropy),
        "attn_varentropy": jnp.mean(attn_varentropy),
        "agreement": jnp.mean(agreement),
        "interaction_strength": interaction_strength
    }

@jax.jit
def new_sample(logits: jax.Array, attention_scores: jax.Array, cur_pos: int, 
               clarifying_question_token: int = 2564, key=jax.random.PRNGKey(1337)) -> jax.Array:
    """Enhanced sampling function that implements the quadrant-based sampling strategy.
    
    This directly implements the framework from the image with four quadrants:
    - Low entropy, low varentropy: greedy sampling
    - High entropy, low varentropy: insert clarifying question
    - Low entropy, high varentropy: explore with temperature
    - High entropy, high varentropy: resample with high temperature
    """
    
    metrics = calculate_metrics(logits, attention_scores)
    ent = metrics["logits_entropy"]
    vent = metrics["logits_varentropy"]
    attn_ent = metrics["attn_entropy"]
    attn_vent = metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    # Thresholds defining the quadrants
    LOW_ENTROPY = 0.01
    HIGH_ENTROPY = 2.1
    LOW_VARENTROPY = 0.05
    HIGH_VARENTROPY = 5.8
    LOW_ATTN_ENTROPY = 11.915
    HIGH_ATTN_ENTROPY = 11.926
    LOW_ATTN_VARENTROPY = 0.001
    HIGH_ATTN_VARENTROPY = 0.009
    LOW_AGREEMENT = 2e-06
    HIGH_AGREEMENT = 5e-06
    LOW_INTERACTION = 0.2
    HIGH_INTERACTION = 0.264

    # Quadrant 1: Low entropy, low varentropy - "flowing with unspoken intent"
    if (ent < LOW_ENTROPY and vent < LOW_VARENTROPY and
        attn_ent < LOW_ATTN_ENTROPY and attn_vent < LOW_ATTN_VARENTROPY and
        agreement < LOW_AGREEMENT and interaction_strength < LOW_INTERACTION):
        return jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)

    # Quadrant 2: High entropy, low varentropy - "treading carefully"
    elif (ent > HIGH_ENTROPY and vent < LOW_VARENTROPY and
          attn_ent < LOW_ATTN_ENTROPY and attn_vent < LOW_ATTN_VARENTROPY and
          agreement < LOW_AGREEMENT and interaction_strength < LOW_INTERACTION):
        return jnp.array([[clarifying_question_token]])

    # Quadrant 3: Low entropy, high varentropy - "exploring forks"
    elif (ent < HIGH_ENTROPY and vent > HIGH_VARENTROPY and
          attn_ent < LOW_ATTN_ENTROPY and attn_vent > HIGH_ATTN_VARENTROPY and
          agreement < LOW_AGREEMENT and interaction_strength > LOW_INTERACTION):
        return _sample(
            logits,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            min_p=0.02,
            key=key
        )

    # Quadrant 4: High entropy, high varentropy - "resampling in the mist"
    elif (ent > HIGH_ENTROPY and vent > HIGH_VARENTROPY and
          attn_ent > HIGH_ATTN_ENTROPY and attn_vent > HIGH_ATTN_VARENTROPY and
          agreement > HIGH_AGREEMENT and interaction_strength > HIGH_INTERACTION):
        return _sample(
            logits,
            temperature=1.2,
            top_p=0.85,
            top_k=27,
            min_p=0.05,
            key=key
        )

    # Default: Adaptive sampling for cases that don't clearly fall into a quadrant
    else:
        return adaptive_sample(
            logits,
            temperature=0.666,
            key=key,
            epsilon=0.1
        )