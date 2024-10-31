from typing import Dict, Tuple
import jax
import jax.numpy as jnp

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

@jax.jit
def multinomial_sample_one(probs_sort: jax.Array, key) -> jax.Array:
    """
    Sample from a multinomial distribution using the Gumbel-max trick.
    This method provides better numerical stability than naive multinomial sampling.
    
    Args:
        probs_sort: Sorted probability distribution array
        key: JAX PRNG key for random number generation
        
    Returns:
        Array of sampled indices with shape [..., 1]
    """
    q = jax.random.exponential(key=key, shape=probs_sort.shape).astype(jnp.bfloat16)
    result = jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)
    return result

@jax.jit
def get_window_probs(sorted_probs: jax.Array, i: jax.Array) -> jax.Array:
    """
    Extract probabilities up to index i from sorted probability distribution.
    Uses efficient masking for TPU optimization.
    
    Args:
        sorted_probs: Sorted probability array of shape [batch_size, vocab_size]
        i: Index up to which probabilities should be included
        
    Returns:
        Masked probability array where values beyond index i are set to zero
    """
    vocab_size = sorted_probs.shape[1]
    indices = jnp.arange(vocab_size, dtype=jnp.int32)
    mask = indices <= i
    mask = jnp.broadcast_to(mask, sorted_probs.shape)
    return jnp.where(mask, sorted_probs, jnp.zeros_like(sorted_probs))

@jax.jit
def basic_sample(logits: jax.Array, *, temperature: float | jax.Array, top_p: float | jax.Array, 
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
def adaptive_sample(logits: jax.Array, *, temperature: float | jax.Array = 0.666, key=jax.random.PRNGKey(1337), epsilon: float = 0.01) -> jax.Array:
    """
    Perform entropy-based adaptive sampling from a probability distribution.
    
    This implementation dynamically determines the set of candidate tokens by measuring
    how each additional token affects the entropy of the sampling distribution. It stops
    adding tokens when their contribution to the entropy falls below a threshold,
    providing an adaptive alternative to fixed top-k or top-p sampling.
    
    Args:
        logits: Raw model logits of shape [batch_size, vocab_size]
        temperature: Softmax temperature to control distribution sharpness (default: 0.666)
        key: JAX PRNG key for sampling (default: fixed seed 1337)
        epsilon: Minimum required entropy gain to include additional tokens (default: 0.01)
        
    Returns:
        Selected token indices of shape [batch_size, 1]
    
    Algorithm:
    1. Convert logits to probabilities using temperature-scaled softmax
    2. Sort tokens by probability in descending order
    3. Iteratively build candidate set:
       - Start with highest probability token
       - Add next token if it increases distribution entropy by >= epsilon
       - Continue until entropy gain falls below epsilon or all tokens processed
    4. Sample from the final candidate distribution
    
    Example:
    For a probability distribution [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01]:
    - First token contribution: -0.5 * log2(0.5) ≈ 0.5 bits
    - Second token adds: -0.3 * log2(0.3) ≈ 0.38 bits
    - Later tokens add progressively less entropy
    - Stop when next token would add < epsilon bits
    
    This approach naturally adapts the sampling pool size based on the shape of
    the probability distribution, avoiding the need for hand-tuned cutoffs.
    """
    bsz = logits.shape[0]
    
    # Cast temperature to bfloat16 and ensure it's an array
    temperature = jnp.array(temperature, dtype=jnp.bfloat16)
    
    # Compute softmax with improved numerical stability
    logits = logits.astype(jnp.bfloat16)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits / temperature)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)
    
    # Use top_k for sorting - very efficient on TPU
    sorted_probs, sorted_indices = jax.lax.top_k(probs, k=probs.shape[-1])
    
    def cond_fn(state):
        current_entropy, previous_entropy, i, mask = state
        entropy_gain = current_entropy - previous_entropy
        return (jnp.any(entropy_gain >= epsilon)) & (i < sorted_probs.shape[-1])
    
    def body_fn(state):
        current_entropy, previous_entropy, i, mask = state
        
        # Get probabilities up to current index
        current_probs = get_window_probs(sorted_probs, i)
        
        # Normalize probabilities
        normalizing_factor = jnp.sum(current_probs, axis=-1, keepdims=True)
        normalized_probs = jnp.where(
            normalizing_factor > 0,
            current_probs / (normalizing_factor + jnp.bfloat16(1e-6)),
            jnp.zeros_like(current_probs)
        )
        
        # Calculate entropy
        log_probs = jnp.log2(jnp.maximum(normalized_probs, jnp.bfloat16(1e-6)))
        new_entropy = -jnp.sum(
            jnp.where(normalized_probs > 0, 
                     normalized_probs * log_probs,
                     jnp.zeros_like(normalized_probs)),
            axis=-1
        )
        
        # Update mask
        entropy_gain = new_entropy - current_entropy
        new_mask = mask.at[:, i].set(entropy_gain >= epsilon)
        
        return (new_entropy, current_entropy, i + 1, new_mask)
    
    # Initialize state
    initial_entropy = jnp.zeros((bsz,), dtype=jnp.bfloat16)
    initial_mask = jnp.zeros((bsz, sorted_probs.shape[-1]), dtype=bool)
    initial_state = (
        initial_entropy,
        initial_entropy,
        jnp.array(0, dtype=jnp.int32),
        initial_mask
    )
    
    # Run the while loop
    final_entropy, _, _, final_mask = jax.lax.while_loop(
        cond_fn, body_fn, initial_state
    )
    
    # Create final candidate distribution
    candidate_probs = jnp.where(final_mask, sorted_probs, jnp.zeros_like(sorted_probs))
    normalizing_factor = jnp.sum(candidate_probs, axis=-1, keepdims=True)
    candidate_probs = candidate_probs / (normalizing_factor + jnp.bfloat16(1e-6))
    
    # Sample and map back to original indices
    next_token = multinomial_sample_one(candidate_probs, key)
    next_token_global = jnp.take_along_axis(
        sorted_indices, 
        next_token.reshape(bsz, 1), 
        axis=-1
    )
    
    return next_token_global.astype(jnp.int32)

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
def calculate_metrics(logits: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Calculate various metrics from logits and attention scores.
    
    This implementation aligns with the quadrant framework shown in the image:
    - Measures both entropy (uncertainty) and varentropy (variance in uncertainty)
    - Tracks attention patterns which can indicate when to use different sampling strategies (removed for now)
    """
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    return {
        "logits_entropy": jnp.mean(entropy),
        "logits_varentropy": jnp.mean(varentropy),
    }

@jax.jit
def new_sample(logits: jax.Array, key=jax.random.PRNGKey(1337), clarifying_question_token: int = 2564) -> jax.Array:
    """Enhanced sampling function that implements the quadrant-based sampling strategy.
    
    This directly implements the framework from the image with four quadrants:
    - Low entropy, low varentropy: greedy sampling
    - High entropy, low varentropy: insert clarifying question
    - Low entropy, high varentropy: explore with temperature
    - High entropy, high varentropy: resample with high temperature
    """
    
    metrics = calculate_metrics(logits)
    ent = metrics["logits_entropy"]
    vent = metrics["logits_varentropy"]
    

    # Thresholds defining the quadrants
    LOW_ENTROPY = 0.01
    HIGH_ENTROPY = 2.1
    LOW_VARENTROPY = 0.05
    HIGH_VARENTROPY = 5.8
    # LOW_ATTN_ENTROPY = 11.915
    # HIGH_ATTN_ENTROPY = 11.926
    # LOW_ATTN_VARENTROPY = 0.001
    # HIGH_ATTN_VARENTROPY = 0.009
    # LOW_AGREEMENT = 2e-06
    # HIGH_AGREEMENT = 5e-06
    # LOW_INTERACTION = 0.2
    # HIGH_INTERACTION = 0.264

    # Quadrant 1: Low entropy, low varentropy - "flowing with unspoken intent"
    if (ent < LOW_ENTROPY and vent < LOW_VARENTROPY):
        return jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)

    # Quadrant 2: High entropy, low varentropy - "treading carefully"
    elif (ent > HIGH_ENTROPY and vent < LOW_VARENTROPY):
        return jnp.array([[clarifying_question_token]])

    # Quadrant 3: Low entropy, high varentropy - "exploring forks"
    elif (ent < HIGH_ENTROPY and vent > HIGH_VARENTROPY):
        return basic_sample(
            logits,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            min_p=0.02,
            key=key
        )

    # Quadrant 4: High entropy, high varentropy - "resampling in the mist"
    elif (ent > HIGH_ENTROPY and vent > HIGH_VARENTROPY):
        return basic_sample(
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