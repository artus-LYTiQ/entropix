from typing import NamedTuple, Dict, Any

class ModelParams(NamedTuple):
  dim: int
  n_layers: int
  n_heads: int
  n_kv_heads: int
  vocab_size: int
  ffn_dim_multiplier: float
  intermediate_size: int
  multiple_of: int
  norm_eps: float
  rope_theta: float
  use_scaled_rope: bool
  max_seq_len: int
  num_devices: int
  model_name: str


class Params(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_theta: float
  use_scaled_rope: bool
  num_devices: int
  model_name: str
  intermediate_size: int


def create_llama_params(params: Dict[str, Any]) -> Params:
  return Params(
    n_layers=params["n_layers"],
    n_local_heads=params["n_heads"],
    n_local_kv_heads=params["n_kv_heads"],
    head_dim=params["dim"] // params["n_heads"],
    max_seq_len=params["max_seq_len"],
    rope_theta=params["rope_theta"],
    use_scaled_rope=params["use_scaled_rope"],
    num_devices=params["num_devices"],
    model_name=params["model_name"],
    intermediate_size=params["intermediate_size"],
  )


# Model configurations
MODEL_1B = ModelParams(
  dim=2048,
  n_layers=16,
  n_heads=32,
  n_kv_heads=8,
  vocab_size=128256,
  ffn_dim_multiplier=1.5,
  intermediate_size=8192,
  multiple_of=256,
  norm_eps=1e-05,
  rope_theta=500000.0,
  use_scaled_rope=True,
  max_seq_len=4096,
  num_devices=1,
  model_name= "Llama_3.2_1B",
)

MODEL_3B = ModelParams(
  dim=3072,
  n_layers=28,
  n_heads=24,
  n_kv_heads=8,
  vocab_size=128256,
  ffn_dim_multiplier=1.5,
  intermediate_size=8192,
  multiple_of=256,
  norm_eps=1e-05,
  rope_theta=500000.0,
  use_scaled_rope=True,
  max_seq_len=4096,
  num_devices=4, # to test mp sharding set to 4 TPUs/GPUs
  model_name= "Llama_3.2_3B",
)

MODEL_70B = ModelParams(
  dim=8192,
  n_layers=80,
  n_heads=64,
  n_kv_heads=8,
  vocab_size=128256,
  ffn_dim_multiplier=1.5,
  intermediate_size=8192,
  multiple_of=256,
  norm_eps=1e-05,
  rope_theta=500000.0,
  use_scaled_rope=True,
  max_seq_len=4096,
  num_devices=8,
  model_name= "Llama_3.2_70B",
)

# Create LLaMA parameters
LLAMA_1B_PARAMS = create_llama_params(MODEL_1B._asdict())
LLAMA_3B_PARAMS = create_llama_params(MODEL_3B._asdict())
LLAMA_70B_PARAMS = create_llama_params(MODEL_70B._asdict())
