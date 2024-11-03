TPU v4-8 has 4 devices with 32 GB RAM each. So an about 20B model is the largest we can expect to fit over all 4 devices. 

# Meshes and model sizes
## Gemini pro
### Mesh Definition and Sharding:

Problem: The current mesh definition in weights.py creates a mesh with only one mp and one fsdp device, effectively using only one core of one TPU. The sharding strategy doesn't distribute the model parameters and computations across the four TPU chips.

Solution: Define a mesh that spans all four TPU chips. For a TPU v4-8, you have 4 chips, each with 2 cores. A common practice is to use a 2x2 or 1x4 mesh for model parallelism (mp) and data parallelism (dp).

import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

#Create a 1x4 mesh (data parallelism over all 4 chips)
devices = mesh_utils.create_device_mesh((1, 4))
mesh = Mesh(devices, ('dp', 'mp'))

with mesh:
    # ... your model code ...

Sharding Strategy: The create_partition_spec function needs to be revised to use the dp axis. A good starting point is to shard the layers across the dp axis: # TODO better understand this

def create_partition_spec(key):
    dp = 'dp'
    mp = 'mp'
    if 'norm' in key or 'rope.freqs' in key:
        return PS()  # Replicate these small parameters
    elif 'tok_embeddings' in key or 'output' in key:
        return PS(mp) # Model parallel for embeddings and output
    elif 'layers' in key:
        layer_num = int(key.split('.')[1]) # Extract layer number
        return PS(dp, mp) # Data and model parallel for layers
    else:
        return PS(mp) # Model parallel for other weights


Flexible Mesh: Since you want to host models of different sizes, a fixed mesh (like the 1x4 proposed earlier) won't be optimal. You need a way to dynamically create meshes based on the model size. Here's an approach:

def create_mesh(model_size_gb):
    num_devices = max(1, int(model_size_gb // 32))  # At least 1 device, scale based on model size
    num_devices = min(num_devices, 4) # Maximum 4 devices (your TPU v4-8 pod) #TODO assert that model does actually fit in 4 devices. This line should not use min() but rather EXCEPT
    devices = mesh_utils.create_device_mesh((1, num_devices))
    return Mesh(devices, ('dp', 'mp'))

4o:
With the new context that this setup is for inference only, a few adjustments and simplifications become apparent. Training requirements, such as gradient synchronization and optimizer states, are irrelevant here, so the focus can be fully on minimizing latency and optimizing memory without needing FSDP’s capabilities for handling optimizer states.
def create_partition_spec(key):
    mp = 'mp'
    
    # Replicate small, frequently accessed parameters
    if 'norm' in key or 'rope.freqs' in key:
        return PS()  # Replicate small params across all devices
    else:
        return PS(mp,)  # Shard large params along 'mp' dimension

# Example usage:
mesh_1b = create_mesh(4) # Llama 1B (approximate size)

### Data Parallelism in Training/Inference:

Problem: The code doesn't explicitly use data parallelism during the xfmr function execution or in the engine's prefill and generate functions. This means you're processing only one example at a time per TPU, severely underutilizing the hardware.

Solution: Use jax.pmap to parallelize the xfmr function over the data parallel axis (dp). You'll also need to replicate the xfmr_weights across the dp axis.

#... in weights.py after loading weights ...
xfmr_weights = jax.device_put_replicated(xfmr_weights, mesh.devices)

#... in model.py ...
p_xfmr = jax.pmap(xfmr, static_broadcasted_argnums=(1,), axis_name='dp', in_axes=(None, None, 0, None, 0, 0, 0))

#... in engine.py or local_main.py ...
logits, kvcache, scores = p_xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis, kvcache, attn_mask)

Batching in prefill and generate: Modify the engine to handle batched inputs. The padded_tokens in prefill and the inputs to generate should have a leading batch dimension.
## Sonnet
def calculate_model_memory_gb(n_params_billions: float) -> float:
    """Calculate model memory in GB (BFloat16)"""
    return n_params_billions * 2  # 2 bytes per parameter

def calculate_runtime_memory_per_batch_gb(
    hidden_dim: int,
    n_layers: int,
    seq_len: int,
    kv_heads: int
) -> float:
    """
    Calculate runtime memory per batch item in GB
    """
    # Activation memory per layer
    activation_memory = (
        4 * hidden_dim * seq_len * 2 +  # QKV projections
        2 * hidden_dim * seq_len * 2 +  # FFN intermediate
        hidden_dim * seq_len * 2        # Layer output
    )
    
    # KV cache memory
    kv_cache_memory = (
        2 * hidden_dim * seq_len * 2 *  # K and V cache
        (kv_heads / (hidden_dim // 64))  # Adjust for grouped-query attention
    )
    
    # Total memory in GB
    total_memory = (
        (activation_memory * n_layers + kv_cache_memory) / 
        (1024 ** 3)  # Convert to GB
    )
    
    return total_memory

1B model
Model weights: 2 GB
Available memory per device: 32 GB - 2 GB = 30 GB
Runtime memory per batch (seq_len=4096):
- Activations: ~1.1 GB
- KV Cache: ~0.5 GB
- Total per batch item: ~1.6 GB

Maximum batch size per device: ~16
Total maximum batch size (4 devices): 64


3B model
Model weights: 6 GB
Available memory per device: 32 GB - 6 GB = 26 GB
Runtime memory per batch (seq_len=4096):
- Activations: ~2.4 GB
- KV Cache: ~0.8 GB
- Total per batch item: ~3.2 GB

Maximum batch size per device: ~7
Total maximum batch size (4 devices): 28


7B model
Model weights: 14 GB
Available memory per device (2x2 mesh): 32 GB - 7 GB = 25 GB
Runtime memory per batch (seq_len=4096):
- Activations: ~4.2 GB
- KV Cache: ~1.0 GB
- Total per batch item: ~5.2 GB

Maximum batch size per device: ~4
Total maximum batch size (4 devices): 16


13B model
Model weights: 26 GB
Available memory per device (1x4 mesh): 32 GB - 6.5 GB = 25.5 GB
Runtime memory per batch (seq_len=4096):
- Activations: ~6.8 GB
- KV Cache: ~1.3 GB
- Total per batch item: ~8.1 GB

Maximum batch size per device: ~2
Total maximum batch size (4 devices): 8


20B model
Model weights: 40 GB
Available memory per device (1x4 mesh): 32 GB - 10 GB = 22 GB
Runtime memory per batch (seq_len=4096):
- Activations: ~8.9 GB
- KV Cache: ~1.5 GB
- Total per batch item: ~10.4 GB

Maximum batch size per device: 1
Total maximum batch size (4 devices): 4


Even larger models do need podslices

# Data types
Optimal Data Types:
You mentioned insisting on bfloat16, which is well-supported by TPUs and recommended for reducing memory consumption. Ensure all tensors and operations are explicitly set to bfloat16 to prevent implicit float32 casting, which may otherwise occur in JAX.


# Memory-mapped Weights:
For loading model weights, your approach of using jnp.load(..., mmap_mode='r', allow_pickle=True) in load_weights() is efficient for single-device loading but does not distribute data evenly across TPUs. Consider using jax.device_put with a custom PartitionSpec during the initial load, ensuring the memory distribution aligns with the TPU layout. For instance:
sharding = NamedSharding(mesh, PS("fsdp", "dp", "mp"))
w[name] = jax.device_put(jnp.load(file), sharding)

# Async multi-threading
## Async TPU-optimized Workload Distribution:
The multi-threaded setup with queues works well for CPU/GPU, but TPU’s runtime benefits more from managing these queues in a non-blocking manner. Use jax.device_put to synchronize prefill and generate threads with TPU mesh management, allowing both inference and backfilling of requests simultaneously without blocking device access.
## Prefill and Generate Threads:
Since TPU tasks can get throttled if threads become idle, ensure max_concurrent_decodes matches the TPU’s hardware concurrency capabilities, potentially adjusting num_engines as needed. For a 4-TPU configuration, starting with a single engine per TPU device, then incrementally scaling up to 2-3 engines per device, can help tune performance without risking OOM issues.


# KVCache
## o1
	•	Dynamic Caching in KVCache:
The KVCache is a crucial part of managing model memory usage, especially for sequences longer than 4,096 tokens. Currently, caching strategies are set up, but tuning may be necessary for high-load TPU deployment:
	•	Increase cache efficiency by managing memory releases explicitly within the KVCache.update method for layers with dense memory (e.g., attention layers).
	•	Enable sharded caching across TPUs by setting up more granular KV cache allocations using partition specifications.
	•	Adjust cache eviction or pruning strategies for long sequences that exceed typical use cases (like conversational models or LLMs), particularly in the free_resource method.
def prune_cache(cache: jax.Array, importance_scores: jax.Array, max_tokens: int):
    """Prune least important tokens from cache when exceeding memory limits."""
    if cache.shape[1] > max_tokens:
        # Keep most important tokens based on attention scores
        _, top_indices = jax.lax.top_k(importance_scores, max_tokens)
        return jax.lax.gather(cache, top_indices, dimensions=(1,))
    return cache

## Gemini
KV Cache Handling:

Problem: The KVCache isn't sharded. This can lead to memory issues and performance bottlenecks, especially with larger batch sizes.

Solution: Shard the KVCache across the dp axis. Each TPU core should store the KV cache for its corresponding batch slice. Modify the KVCache.new and KVCache.update functions accordingly.

# Potential Issues
## o1 and Sonnet
	•	Unused Argument: The sampler parameter is not used. If it’s intended to allow custom sampling strategies, the code should incorporate it.
	•	Batch Dimension Handling: Ensure that all tensors handle batch dimensions correctly, especially if generating tokens for multiple sequences in parallel.
	•	Reduce Sampling Loops in adaptive_sampling:
The current implementation samples repeatedly, which could lead to longer runtimes. TPU-optimized sampling may benefit from a single-pass sampling setup (e.g., batching adaptive samples into larger chunks). Consider refining by removing high-cost scoring functions, such as confidence_score, or by reducing adaptive sampling parameters.
	•	Use Vectorized Sampling:
Vectorizing multinomial_sample_one and nucleus_sample to operate over the entire batch instead of iteratively sampling can improve performance by reducing TPU idle times. Additionally, increasing the batch size or padding smaller batches to match TPU hardware parallelism may further increase sampling efficiency.
	•	Batch Dimension Handling: The metrics (ent, vent, attn_ent, attn_vent) are computed as means, resulting in scalar values. However, when applying logical operations, they might not align correctly with batch dimensions if processing multiple sequences.
	•	Case Selection Logic:
	•	The conditions are converted to floats and stacked, and jnp.argmax is used to select the case. This might not be robust, especially if multiple conditions evaluate to True (since they are cast to 1.0).
	•	Using jnp.argmax in this way could lead to unexpected behavior if more than one condition is True. A better approach might be to check conditions in order of priority or use a more explicit case selection mechanism.
	•	Undefined Variables:
	•	In the helv function, there’s a commented-out section mentioning gen_tokens being undefined. This indicates incomplete implementation or a missing context.
	•	Clarifying Question Logic:
	•	The helv function always returns the clarifying_question_token, which might not be appropriate if this token has already been generated recently. The commented-out code suggests an intention to avoid repetitive questions, but it’s incomplete.
	•	_and function: Note: Since JAX operations are used, ensure that inputs are compatible (e.g., they are JAX arrays).


Potential Optimizations
	•	Batch Processing:
	•	Ensure that all computations correctly handle batch dimensions to support generating tokens for multiple sequences simultaneously.
	•	Case Selection Improvement:
	•	Refactor the case selection to handle multiple True conditions more explicitly.
	•	Consider assigning priorities to conditions or combining conditions where appropriate.
	•	Complete the helv Function:
	•	Implement logic to check if the clarifying question has already been asked recently, to avoid repetition.
	•	Define or pass gen_tokens if needed.

4. Potential Flaws and Areas for Optimization

4.1. Incomplete or Undefined Variables

	•	Undefined gen_tokens: In the helv function, gen_tokens is referenced in the commented-out code but is not defined within the scope. This needs to be addressed for the function to work correctly.
	•	Unused sampler Argument: The generate function has a sampler parameter that’s not used. If the intention is to allow custom sampling strategies, the code should be updated to utilize this argument.

4.2. Batch Dimension Handling

	•	Scalar vs. Batch Metrics: Metrics are computed as means, resulting in scalar values. If the model processes batches, metrics should be computed per sample in the batch to adapt sampling strategies individually.
	•	Logical Operations with Scalars: When applying conditions, ensure that scalar metrics are appropriately compared to thresholds, and that the resulting boolean masks align with batch dimensions.

4.3. Case Selection Logic

	•	Ambiguity in Conditions: Multiple conditions might evaluate to True simultaneously, but using jnp.argmax on the stacked conditions assumes only one will be True.
	•	Switch Function Limitations: The use of jax.lax.switch with the selected case might not handle batch-wise different cases. If each sample in the batch might satisfy a different condition, this approach won’t work as intended.

4.4. Sampling Strategy Adaptability

	•	Adaptive Sampling Complexity: The adaptive sampling strategy involves multiple resamplings and scoring, which could be computationally expensive.
	•	Scoring Function: The score_sample function uses a combination of log probabilities and confidence scores. Ensure that this scoring adequately reflects the desired qualities of the samples.

4.5. Code Readability and Maintenance

	•	Hardcoded Constants: Some constants, like MAX_K, are hardcoded. Consider parameterizing them or documenting their purpose.
	•	Code Comments and Documentation: Add more comments explaining the purpose of each section, especially in complex functions like sample.

## Gemini
Communication: Communication between TPU cores can be a bottleneck. Optimize your sharding strategy to minimize communication overhead.

Compilation Time: XLA compilation can take time. Minimize recompilation by using static shapes and avoiding dynamic control flow whenever possible.

# Recommendations
## o1-preview recommendations
	1.	Batch-wise Condition Handling:
	•	Modify the metric calculations and condition checks to operate per sample in the batch.
	•	Use vectorized operations to determine the sampling strategy for each sequence individually.
	2.	Improve Case Selection Logic:
	•	Instead of stacking conditions and using jnp.argmax, iterate over conditions with explicit checks.
	•	Consider using jax.lax.cond or jax.lax.switch with more sophisticated logic to handle multiple True conditions.
	3.	Complete Incomplete Functions:
	•	Address the FIXME in the helv function by defining gen_tokens or refactoring the code to avoid its use.
	•	Ensure that all variables used within functions are properly defined and scoped.
	4.	Optimize Adaptive Sampling:
	•	Assess the computational cost of adaptive sampling with multiple resamples.
	•	Consider reducing the number of adaptive samples or optimizing the scoring function for efficiency.
	5.	Refactor and Modularize Code:
	•	Break down complex functions into smaller, reusable components.
	•	Parameterize hardcoded values and provide clear documentation for each parameter.
	6.	Test and Validate Sampling Strategies:
	•	Implement unit tests to verify that each sampling strategy behaves as expected.
	•	Evaluate the impact of different sampling strategies on generated text quality.
	7.	Utilize sampler Argument:
	•	Update the generate function to accept a custom sampler if provided.
	•	This allows for greater flexibility and easier experimentation with different sampling methods.

## Gemini recommendations
4. Sampler Function:

Problem: Not pmapped.

Solution: pmap the sampler function as it is called from within the pmapped xfmr function.

@functools.partial(jax.pmap, static_broadcasted_argnums=(0, 1), axis_name='dp', in_axes=(None, None, 0, None, None, None))
def prefill(self, params, existing_prefix, padded_tokens, true_length, sampler, rng):
    # ... your existing prefill logic ...

5. Orchestrator and Engine Changes for Batching:

The EntropixOrchestrator and EntropixEngine will need modifications to support batching. This includes changes to the prefill, generate, and decode methods to handle batched inputs and outputs.

6. Model Saving and Loading:

Save the sharded model weights to disk. This will significantly speed up loading times, as you won't have to reshard the model every time you start your application. Use numpy.save with the mmap_mode='r' option for efficient memory-mapped loading.

## docs
vmap is tracing the function and automatically adding batch axes at the beginning of each input.
jax.lax.map(), which is a sequential map rather than a vectorization

Use jax.debug.breakpoint() to pause the execution of your JAX program to inspect values
https://jax.readthedocs.io/en/latest/_tutorials/advanced-debugging.html#advanced-debugging
For value-dependent breakpointing, you can use runtime conditionals like jax.lax.cond() or jax.lax.switch() to control when the breakpoint is hit.
the more flexible jax.debug.callback()

use the new-style typed PRNG keys produced by jax.random.key(), rather than the old-style raw PRNG keys produced by jax.random.PRNGKey()!

key, subkey = random.split(key)
key, *forty_two_subkeys = random.split(key, num=43)

If you shard the leading axis of both x and weights in the same way, then the matrix multiplication will automatically happen in parallel

JAX transformations like jit(), vmap(), grad(), require the functions they wrap to be pure: that is, functions whose outputs depend solely on the inputs, and which have no side effects such as updating of global state.