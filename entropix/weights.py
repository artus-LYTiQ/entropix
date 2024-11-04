from typing import List, NamedTuple, Optional
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
from jax.experimental import mesh_utils
from pathlib import Path
from contextlib import nullcontext

from entropix.config import Params

class LayerWeights(NamedTuple):
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    w1: jax.Array
    w2: jax.Array
    w3: jax.Array
    ffn_norm: jax.Array
    attention_norm: jax.Array

class XfmrWeights(NamedTuple):
    tok_embeddings: jax.Array
    norm: jax.Array
    output: jax.Array
    layer_weights: List[LayerWeights]

# Define sharding specifications
def create_partition_spec(key, num_devices):
  """
  Initially, we only use a very simple partitioning: 
  a.) No partitioning for models that fit on 1 device. 
  b.) Sharding the larger weights over the devices for larger models.
  """
  if num_devices == 1:
      return PS()  # No sharding needed for single device

  # Replicate certain parameters without sharding
  if "norm" in key or "rope.freqs" in key:
      return PS()  # Replicated parameters
  
  # Model parallel for larger matrices based on key name
  #elif any(layer in key for layer in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]):
  return PS("x")

def load_weights(ckpt_dir: Path, model_params: Params, mesh: Optional[Mesh]):
    w = {}
    layer_weights = []

    num_devices = len(mesh.devices) if mesh is not None else 1

    context = mesh if mesh is not None else nullcontext()

    # Load weights
    with context:
      # Load weights
      for file in ckpt_dir.glob("*.npy"):
          name = ".".join(str(file).split("/")[-1].split(".")[:-1])
          weight = jnp.load(file=file, mmap_mode="r", allow_pickle=True)

          partition_spec = create_partition_spec(name, num_devices)
          if mesh is not None:
              sharding = NamedSharding(mesh, partition_spec)
              weight = jax.device_put(weight, sharding)
          else:
              weight = jax.device_put(weight)  # Single device

          w[name] = weight
          #print(f"{name} placed on:", weight.device if mesh is None else weight.sharding)

          # Organize weights into structured format for model layers
          if any(lyr in name for lyr in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]):
            weight = weight.T
            if "wq" in name or "wk" in name or "wv" in name:
              weight = weight.reshape(
                -1,
                model_params.n_local_heads if "wq" in name else model_params.n_local_kv_heads,
                model_params.head_dim,
              )
          w[name] = weight
          #print(f"{name} placed on:", weight.device if num_devices == 1 else weight.sharding)

      for i in range(model_params.n_layers):
        layer_weights.append(
          LayerWeights(
            wq=w[f"layers.{i}.attention.wq.weight"],
            wk=w[f"layers.{i}.attention.wk.weight"],
            wv=w[f"layers.{i}.attention.wv.weight"],
            wo=w[f"layers.{i}.attention.wo.weight"],
            w1=w[f"layers.{i}.feed_forward.w1.weight"],
            w2=w[f"layers.{i}.feed_forward.w2.weight"],
            w3=w[f"layers.{i}.feed_forward.w3.weight"],
            ffn_norm=w[f"layers.{i}.ffn_norm.weight"],
            attention_norm=w[f"layers.{i}.attention_norm.weight"],
          )
        )

    # Finalize structured weights for XfmrWeights
    xfmr_weights = XfmrWeights(
      tok_embeddings=w["tok_embeddings.weight"],
      norm=w["norm.weight"],
      output=w["output.weight"],
      layer_weights=layer_weights,
    )

    return xfmr_weights, model_params