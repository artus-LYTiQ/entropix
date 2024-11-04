import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from pathlib import Path
from typing import OrderedDict

from entropix.engine import EntropixEngine
from entropix.weights import load_weights, create_partition_spec, XfmrWeights
from entropix.config import Params, LLAMA_1B_PARAMS, LLAMA_3B_PARAMS
from entropix.tokenizer import Tokenizer
from entropix.model import xfmr
from entropix.orchestrator import Driver, EntropixOrchestrator
from entropix.sampler import sample

@pytest.fixture
def setup_single_device_mesh():
    # Single device configuration for test
    return None  # None represents single device setup in JAX

@pytest.fixture
def setup_multi_device_mesh():
    # Multi-device (4 devices in a TPU configuration)
    devices = jax.devices()[:4]
    mesh = Mesh(devices, ["x"])
    return mesh

def test_single_device_partition_spec():
    # Test partition spec with a single device setup
    spec = create_partition_spec("norm", num_devices=1)
    assert spec == PS()  # No partitioning for single device

def test_multi_device_partition_spec():
    # Test partition specs with a multi-device setup
    spec_embedding = create_partition_spec("tok_embeddings", num_devices=4)
    spec_output = create_partition_spec("output", num_devices=4)
    spec_attention = create_partition_spec("wq", num_devices=4)
    assert spec_embedding == PS("x")  # Data and model parallel for embeddings
    assert spec_output == PS("x")
    assert spec_attention == PS("x")  # Model parallel for larger matrices

def weight_load_helper(params, ckpt_path, i):
    # Determine devices for this model
    devices = jax.devices()[i * params.num_devices : (i + 1) * params.num_devices]

    # Create mesh if more than one device is used per model
    if params.num_devices > 1:
        mesh = Mesh(devices, ("x",))
        context = mesh
    else:
        mesh = None  # Single-device case
        context = jax.default_device(devices[0])

    # Load weights using the provided mesh
    with context:
        xfmr_weights, _ = load_weights(ckpt_path, params, mesh)
        assert xfmr_weights is not None

        if params.num_devices == 1:
            assert mesh is None
        else:
            assert isinstance(mesh, Mesh)   
            assert mesh.shape == OrderedDict([("x", params.num_devices)])
    return mesh,xfmr_weights

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

    

def test_load_weights_single_device():
    # Load weights in a single device setup
    ckpt_path: Path = Path("weights/1B-Instruct")  
    i = 0  # Index for device selection
    mesh, xfmr_weights = weight_load_helper(LLAMA_1B_PARAMS, ckpt_path, i)
    validate_xfmr_weights(xfmr_weights, LLAMA_1B_PARAMS)
    print(f"{xfmr_weights.tok_embeddings.shape=}") 

def test_load_weights_multi_device():
    # Load weights in a multi-device setup
    ckpt_path = Path("weights/3B-Instruct")  
    i = 0  # Index for device selection

    # Create mesh if more than one device is used per model
    mesh, xfmr_weights = weight_load_helper(LLAMA_3B_PARAMS, ckpt_path, i)
    print(mesh.devices, mesh.axis_names, mesh.shape)  # Print mesh details for debugging
    validate_xfmr_weights(xfmr_weights, LLAMA_3B_PARAMS)

def test_engine_initialization_single_device(setup_single_device_mesh):
    # Initialize engine with single device
    xfmr_weights = test_load_weights_single_device()  # Load weights for single device
    tokenizer_path: str = "entropix/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)
    engine = EntropixEngine(LLAMA_1B_PARAMS, xfmr_weights, setup_single_device_mesh, tokenizer, None, None)
    assert engine.mesh == setup_single_device_mesh  # Should match single device (None)

def test_engine_initialization_multi_device(setup_multi_device_mesh):
    # Initialize engine with multi-device
    xfmr_weights = test_load_weights_multi_device()  # Load weights for multi-device
    tokenizer_path: str = "entropix/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)
    engine = EntropixEngine(LLAMA_3B_PARAMS, xfmr_weights, setup_multi_device_mesh, tokenizer, None, None)
    assert engine.mesh == setup_multi_device_mesh  # Should match multi-device mesh


def test_inference_single_device(setup_single_device_mesh):
    # Initialize engine with single device
    _, xfmr_weights = weight_load_helper(LLAMA_1B_PARAMS, Path("weights/1B-Instruct"), i=0)  # Load weights for single device
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
    _, xfmr_weights = weight_load_helper(LLAMA_1B_PARAMS, Path("weights/1B-Instruct"), i=0)  # Load weights for single device
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