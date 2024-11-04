import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from pathlib import Path
from typing import OrderedDict

from entropix.engine import EntropixEngine
from entropix.weights import load_weights, XfmrWeights
from entropix.config import Params, LLAMA_1B_PARAMS, LLAMA_3B_PARAMS
from entropix.tokenizer import Tokenizer
from entropix.model import xfmr
from entropix.orchestrator import Driver, EntropixOrchestrator
from entropix.sampler import sample
from entropix.engine_main import setup_engine_mesh_and_partition_spec

@pytest.fixture(params=["single_engine", "multi_engine"])
def engine_test_type(request):
    """Fixture that indicates whether the test is for a single engine or multiple engines."""
    return request.param

@pytest.fixture
def setup_device_or_mesh(engine_test_type):
    """Fixture that sets up either a single device or a multi-device mesh based on the engine test type."""
    
    if engine_test_type == "single_engine":
        engine_meshes, engine_partition_specs = setup_engine_mesh_and_partition_spec(total_devices=4, num_devices_per_engine=1, num_engines=4)
        is_multi_device = False

    elif engine_test_type == "multi_engine":
        engine_meshes, engine_partition_specs = setup_engine_mesh_and_partition_spec(total_devices=4, num_devices_per_engine=4, num_engines=1)
        is_multi_device = True

    return {"meshes": engine_meshes, "specs": engine_partition_specs, "multi" : is_multi_device}

def validate_xfmr_weights(xfmr_weights: XfmrWeights, model_params: Params):
    # Basic checks for main fields in XfmrWeights
    assert xfmr_weights.tok_embeddings is not None, "tok_embeddings is None"
    assert xfmr_weights.norm is not None, "norm is None"
    assert xfmr_weights.output is not None, "output is None"
    assert xfmr_weights.layer_weights, "layer_weights is empty or None"

    # Expected shapes based on model parameters
    vocab_size = 128256
    dim = model_params.head_dim * model_params.n_local_heads
    n_heads = model_params.n_local_heads
    n_kv_heads = model_params.n_local_kv_heads
    head_dim = model_params.head_dim
    ffn_dim = int(dim * 1.5)
    intermediate_size = model_params.intermediate_size

    # Validate tok_embeddings, norm, and output shapes
    assert xfmr_weights.tok_embeddings.shape == (vocab_size, dim), f"Unexpected shape for tok_embeddings: {xfmr_weights.tok_embeddings.shape}"
    assert xfmr_weights.norm.shape == (dim,), f"Unexpected shape for norm: {xfmr_weights.norm.shape}"
    assert xfmr_weights.output.shape == (vocab_size, dim), f"Unexpected shape for output: {xfmr_weights.output.shape}"

    # Validate each layer in layer_weights
    for i, layer in enumerate(xfmr_weights.layer_weights):
        print(f"Validating layer {i}")

        # Expected shapes for attention weights
        expected_shape_attn = (dim, n_heads, head_dim)  # Applies to wq, wk, wv, wo
        expected_shape_ffn = (dim, ffn_dim)  # Applies to w1, w2, w3

        # Attention weights
        assert layer.wq.shape == (dim, n_heads, head_dim), f"Unexpected shape for layer {i} wq: {layer.wq.shape}"
        assert layer.wk.shape == (dim, n_kv_heads, head_dim), f"Unexpected shape for layer {i} wk: {layer.wk.shape}"
        assert layer.wv.shape == (dim, n_kv_heads, head_dim), f"Unexpected shape for layer {i} wv: {layer.wv.shape}"
        assert layer.wo.shape == (dim, dim), f"Unexpected shape for layer {i} wo: {layer.wo.shape}"

        # Feedforward network weights
        assert layer.w1.shape == (dim, intermediate_size), f"Unexpected shape for layer {i} w1: {layer.w1.shape}" # intermediate_size
        assert layer.w2.shape == (intermediate_size, dim), f"Unexpected shape for layer {i} w2: {layer.w2.shape}" # intermediate_size
        assert layer.w3.shape == (dim, intermediate_size), f"Unexpected shape for layer {i} w3: {layer.w3.shape}" # intermediate_size

        # Norm layers
        assert layer.ffn_norm.shape == (dim,), f"Unexpected shape for layer {i} ffn_norm: {layer.ffn_norm.shape}"
        assert layer.attention_norm.shape == (dim,), f"Unexpected shape for layer {i} attention_norm: {layer.attention_norm.shape}"

        # Optionally, log summaries for each component
        #print(f"Layer {i} wq: mean={jnp.mean(layer.wq)}, min={jnp.min(layer.wq)}, max={jnp.max(layer.wq)}")
        #print(f"Layer {i} wk: mean={jnp.mean(layer.wk)}, min={jnp.min(layer.wk)}, max={jnp.max(layer.wk)}")
        #print(f"Layer {i} wv: mean={jnp.mean(layer.wv)}, min={jnp.min(layer.wv)}, max={jnp.max(layer.wv)}")
        # Repeat for other components as needed

def test_load_weights(setup_device_or_mesh):
    """Test weight loading with single or multiple engines, adjusting based on the engine type."""
    meshes = setup_device_or_mesh["meshes"]
    specs = setup_device_or_mesh["specs"]
    multi_device = setup_device_or_mesh["multi"]

    # Call load_weights with the correct params and either mesh or device
    device = None
    if multi_device:
        weights = load_weights(ckpt_dir=Path("weights/3B-Instruct"), model_params=LLAMA_3B_PARAMS, mesh=meshes[0], partition_spec=specs[0], device=device)
    else:
        device=jax.local_devices()[0] # just take first device for now. will work until we look over all possible engines
        weights = load_weights(ckpt_dir=Path("weights/1B-Instruct"), model_params=LLAMA_1B_PARAMS, mesh=meshes[0], partition_spec=specs[0], device=device)

    # Validate the weights based on single-engine or multi-engine setup
    if meshes[0]:
        # Multi-device case: verify weights are sharded across mesh devices
        assert all(weight.is_sharded for weight in weights.values()), "Weights should be sharded for multi-device mesh"
        # print out the sharding information
        for weight in weights.values():
            print(f"Weight sharding: {weight.sharding}")
    else:
        # Single-device case: verify weights are placed on the specified device
        for weight in weights.values():
            assert weight.device() == device, f"Weight should be on device {device}"

def test_engine_initialization_single_device(setup_single_device_mesh):
    # Initialize engine with single device
    xfmr_weights = TODO  # Load weights for single device
    tokenizer_path: str = "entropix/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)
    engine = EntropixEngine(LLAMA_1B_PARAMS, xfmr_weights, setup_single_device_mesh, tokenizer, None, None)
    assert engine.mesh == setup_single_device_mesh  # Should match single device (None)

def test_engine_initialization_multi_device(setup_multi_device_mesh):
    # Initialize engine with multi-device
    xfmr_weights = TODO  # Load weights for multi-device
    tokenizer_path: str = "entropix/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)
    engine = EntropixEngine(LLAMA_3B_PARAMS, xfmr_weights, setup_multi_device_mesh, tokenizer, None, None)
    assert engine.mesh == setup_multi_device_mesh  # Should match multi-device mesh


def test_inference_single_device(setup_single_device_mesh):
    # Initialize engine with single device
    _, xfmr_weights = TODO (LLAMA_1B_PARAMS, Path("weights/1B-Instruct"), i=0)  # Load weights for single device
    assert xfmr_weights.tok_embeddings is not None, "Token embeddings missing"
    tokenizer_path: str = "entropix/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)
    engine = EntropixEngine(LLAMA_1B_PARAMS, xfmr_weights, setup_single_device_mesh, tokenizer, xfmr_fn=jax.jit(xfmr, static_argnames=("model_params",)),
          sample_fn=jax.jit(sample),)
    decode_state = {
        "next_pos": 0,
        "tokens": jax.numpy.array([[0]]),
        "generated_tokens": jax.numpy.zeros((1, 1), dtype=jax.numpy.int32),
        "cache": jax.numpy.zeros((1, 1, 1, 1))  # Replace with actual cache structure if necessary
    }
    result = engine.generate(LLAMA_1B_PARAMS, decode_state=decode_state, rng=jax.random.PRNGKey(0))
    assert result is not None

def test_inference_multi_device(setup_multi_device_mesh):
    # Initialize engine with single device
    _, xfmr_weights = TODO (LLAMA_1B_PARAMS, Path("weights/1B-Instruct"), i=0)  # Load weights for single device
    tokenizer_path: str = "entropix/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)
    engine = EntropixEngine(LLAMA_3B_PARAMS, xfmr_weights, setup_multi_device_mesh, tokenizer, xfmr_fn=jax.jit(xfmr, static_argnames=("model_params",)),
          sample_fn=jax.jit(sample),)
    decode_state = {
        "next_pos": 0,
        "tokens": jax.numpy.array([[0]]),
        "generated_tokens": jax.numpy.zeros((1, 1), dtype=jax.numpy.int32),
        "cache": jax.numpy.zeros((1, 1, 1, 1))  # Replace with actual cache structure if necessary
    }
    result = engine.generate(LLAMA_3B_PARAMS, decode_state=decode_state, rng=jax.random.PRNGKey(0))
    assert result is not None