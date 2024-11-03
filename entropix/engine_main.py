
import asyncio
from pathlib import Path

import jax
import tyro

import os

from entropix.engine import LLAMA_1B_PARAMS, EntropixEngine
from entropix.model import xfmr
from entropix.orchestrator import Driver, EntropixOrchestrator
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights

from jax.experimental import mesh_utils
from jax.sharding import Mesh

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


async def run(
  ckpt_path: Path = Path("weights/1B-Instruct"),
  tokenizer_path: str = "entropix/tokenizer.model",
  num_devices_per_model: int = 1,
):
  model_params = LLAMA_1B_PARAMS

  total_devices = jax.device_count()
  num_models = total_devices // num_devices_per_model
  if total_devices % num_devices_per_model != 0:
      print(f"Warning: {total_devices % num_devices_per_model} devices will be unused.")

  # In case of 1 physical device (TPU or GPU), we use jax.default_device.
  # In case of multiple devices, we use jax.sharding.Mesh for multi-device cases.
  engines = []
  for i in range(num_models):
      if num_devices_per_model == 1:  # Single device case
          device = jax.local_devices()[i]
          with jax.default_device(device):
              xfmr_weights, _ = load_weights(ckpt_path, model_params)
              mesh = None
      else:  # Multi-device case - MP sharding
          devices = jax.local_devices()[i * num_devices_per_model:(i + 1) * num_devices_per_model]
          mesh = Mesh(devices, ("mp",))
          with mesh:
              xfmr_weights, _ = load_weights(ckpt_path, model_params._replace(num_devices=num_devices_per_model))

      # Engine creation
      with mesh if mesh else jax.default_device(device if num_devices_per_model==1 else jax.devices()[0]): # include else jax.devices()[0] for cases where there are only TPUs and no devices are assigned yet
          tokenizer = Tokenizer(tokenizer_path)
          xfmr_fn = jax.jit(xfmr, static_argnames=("model_params",))
          sample_fn = jax.jit(sample)

          engine = EntropixEngine(
              model_params if num_devices_per_model == 1 else model_params._replace(num_devices=num_devices_per_model),
              xfmr_weights,
              mesh,
              tokenizer,
              xfmr_fn,
              sample_fn,
          )
          engines.append(engine)

  driver = Driver(
      prefill_engines=engines,
      generate_engines=engines,  # You'll likely want separate generate engines in a real application
      prefill_params=[model_params if num_devices_per_model == 1 else model_params._replace(num_devices=num_devices_per_model)] * num_models,
      generate_params=[model_params if num_devices_per_model == 1 else model_params._replace(num_devices=num_devices_per_model)] * num_models,
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


# Check for GPU availability *before* setting XLA flags
if jax.devices("gpu"):  # Check if any GPUs are available
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
    print("GPU detected. Setting XLA flags for GPU optimization.")
elif jax.devices("tpu"):
    print("TPU detected. No GPU-specific XLA flags set.") # Indicate TPU usage
else:
    print("WARNING: No GPU or TPU detected. Using CPU.")

if __name__ == "__main__":
  tyro.cli(main)
