
import asyncio
from pathlib import Path

import jax
import tyro

import os

from entropix.engine import EntropixEngine
from entropix.model import xfmr
from entropix.orchestrator import Driver, EntropixOrchestrator
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights

from entropix.config import LLAMA_1B_PARAMS

from jax.experimental import mesh_utils
from jax.sharding import Mesh

from contextlib import nullcontext

class Metadata:
  def __init__(self):
    self.start_time = None


class Request:
  def __init__(
    self,
    tokens: jax.Array,
    max_tokens: int,
    metadata: Metadata,
    is_client_side_tokenization: bool = False,
  ):
    self.tokens: jax.Array = tokens
    self.max_tokens: int = max_tokens
    self.metadata: Metadata = metadata
    self.is_client_side_tokenization: bool = is_client_side_tokenization

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as PS

def setup_engine_mesh_and_partition_spec(total_devices, num_devices_per_engine, num_engines):
    """Configures mesh and partition specs for each engine based on device requirements."""
    devices = jax.devices()
    
    engine_meshes = []
    engine_partition_specs = []
    
    for i in range(num_engines):
        # Determine devices for the current engine
        start_idx = i * num_devices_per_engine
        engine_devices = devices[start_idx: start_idx + num_devices_per_engine]
        
        # Case 1: Single device per engine (no mesh needed)
        if num_devices_per_engine == 1:
            mesh = None  # No mesh for single-device engines
            partition_spec = {
                "attention": PS(),         # No partitioning
                "ffn": PS(),               # No partitioning
                "embedding": PS()
            }
        
        # Case 2: Multi-device per engine (use a mesh and partition specs)
        else:
            mesh = Mesh(engine_devices, ("x",))  # Define a 1D mesh for multi-device sharding
            partition_spec = {
                "attention": PS("x"),      # Shard attention heads across devices
                "ffn": PS("x"),            # Shard feed-forward network layers
                "embedding": PS("x")       # Shard embeddings if they are large
            }
        
        # Append the mesh and partition spec for the current engine
        engine_meshes.append(mesh)
        engine_partition_specs.append(partition_spec)

    print(f"Configured {num_engines} engines with {num_devices_per_engine} devices per engine.")
    return engine_meshes, engine_partition_specs

async def run(
  ckpt_path: Path = Path("weights/1B-Instruct"),
  tokenizer_path: str = "entropix/tokenizer.model",
):
  
  model_params = LLAMA_1B_PARAMS
  print(f"Loading model {model_params.model_name} from path {ckpt_path}")

  total_devices = jax.device_count()
  num_devices_per_model = model_params.num_devices
  num_engines = total_devices // num_devices_per_model

  print(f"Total devices: {total_devices}")
  print(f"Devices per model: {num_devices_per_model}")
  print(f"Number of models: {num_engines}")

  if total_devices % num_devices_per_model != 0:
      print(f"Warning: {total_devices % num_devices_per_model} devices will be unused.")

  # In case of 1 physical device (TPU or GPU), we use jax.default_device.
  # In case of multiple devices, we use jax.sharding.Mesh for multi-device cases.
  engines = []
  engine_meshes, partition_specs = setup_engine_mesh_and_partition_spec(
    total_devices, num_devices_per_model, num_engines
)

  for i in range(num_engines):
      # Load weights using the provided mesh
      context = engine_meshes[i] if engine_meshes[i] else nullcontext()

      with context:
        xfmr_weights, _ = load_weights(ckpt_path, model_params, mesh=context, partition_spec=partition_specs[i], device=jax.local_devices[i])

      # Create tokenizer
      tokenizer = Tokenizer(tokenizer_path)

      # Create engine
      engine = EntropixEngine(
          model_params,
          xfmr_weights,
          mesh,
          tokenizer,
          xfmr_fn=jax.jit(xfmr, static_argnames=("model_params",)),
          sample_fn=jax.jit(sample),
      )
      engines.append(engine)

  driver = Driver(
      prefill_engines=engines,
      generate_engines=engines,  # You'll likely want separate generate engines in a real application
      #prefill_params=[model_params if num_devices_per_model == 1 else model_params._replace(num_devices=num_devices_per_model)] * num_models,
      #generate_params=[model_params if num_devices_per_model == 1 else model_params._replace(num_devices=num_devices_per_model)] * num_models,
      prefill_params=[model_params] * num_engines,
      generate_params=[model_params] * num_engines,
  )

  orchestrator = EntropixOrchestrator(driver)
  prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

Think carefully in a step-by-step manner. which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

  # Think carefully in a step-by-step manner. Can you write a python agent that generates passwords with modern best practices?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  # Think carefully in a step-by-step manner. Oliver picks 44 kiwis on Friday. Then he picks 58 kiwis on Saturday. On Sunday, he picks double the number of kiwis he did on Friday, but five of them were a bit smaller than average. How many kiwis does Oliver have?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  # Think carefully in a step-by-step manner. which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

  # Create separate request instances but process them through the same engines
  requests = [
    Request(tokens=prompt, max_tokens=4096, metadata=Metadata()) for _ in range(4)
  ]

  # Process requests concurrently through the shared engines
  generators = [orchestrator.decode(request) for request in requests]

  async def process_generator(gen, loop_num):
    async for decoded in gen:
      print(f"LOOP {loop_num}: {decoded}")

  await asyncio.gather(
    *[process_generator(gen, i + 1) for i, gen in enumerate(generators)]
  )


def main():
  asyncio.run(run())


# Check available platforms
available_platforms = {device.platform for device in jax.devices()}

if "gpu" in available_platforms:  # Set flags if GPUs are available
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
    print("GPU detected. Setting XLA flags for GPU optimization.")
elif "tpu" in available_platforms:
    print("TPU detected. No GPU-specific XLA flags set.")  # Indicate TPU usage
else:
    print("WARNING: No GPU or TPU detected. Using CPU.")

if __name__ == "__main__":
  tyro.cli(main)
