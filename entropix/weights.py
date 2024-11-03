from typing import List, NamedTuple

import jax
import jax.numpy as jnp

from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
from jax.experimental import mesh_utils

from pathlib import Path

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


def create_partition_spec(key, num_devices):
  dp = "dp"
  mp = "mp"
  fsdp = "fsdp" # not used yet

  if num_devices == 1:  # Single device case
    return PS()  # No sharding needed

  if "norm" in key or "rope.freqs" in key:
    return PS() # Replicated parameters

  elif "tok_embeddings" in key or "output" in key:
    return PS(dp, mp) # Data parallel for embeddings and output layer
  elif "w2" in key or "wo" in key:
    return PS(mp, dp) # Model parallel for larger matrices
  else:
    return PS(dp, mp)  # Data and model parallel for other layers


def load_weights(ckpt_dir: Path, model_params: Params):
  w = {}
  layer_weights = []

  num_devices = model_params.num_devices
  if num_devices > 1:
    mesh_shape = mesh_utils.create_device_mesh((num_devices, 1))
    mesh = Mesh(mesh_shape, ("dp", "mp"))
  else:
    mesh = None

  with jax.default_device(jax.devices()[0]) if mesh is None else mesh:
    for file in ckpt_dir.glob("*.npy"):
      name = ".".join(str(file).split("/")[-1].split(".")[:-1])
      weight = jnp.load(file=file, mmap_mode="r", allow_pickle=True)
      partition_spec = create_partition_spec(name, num_devices)

      if mesh is not None:
        sharding = NamedSharding(mesh, partition_spec)
        weight = jax.device_put(weight, sharding)
      else:
        weight = jax.device_put(weight)

      if any(lyr in name for lyr in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]):
        weight = weight.T
        if "wq" in name or "wk" in name or "wv" in name:
          weight = weight.reshape(
            -1,
            model_params.n_local_heads if "wq" in name else model_params.n_local_kv_heads,
            model_params.head_dim,
            )
      w[name] = weight


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

    xfmr_weights = XfmrWeights(
      tok_embeddings=w["tok_embeddings.weight"],
      norm=w["norm.weight"],
      output=w["output.weight"],
      layer_weights=layer_weights,
    )

  return xfmr_weights, mesh