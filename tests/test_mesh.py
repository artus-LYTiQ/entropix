import pytest
import jax
from jax.sharding import Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from pathlib import Path
from typing import OrderedDict

from entropix.engine import EntropixEngine
from entropix.weights import load_weights, create_partition_spec
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

def test_load_weights_single_device():
    # Load weights in a single device setup
    ckpt_path: Path = Path("weights/1B-Instruct")  
    i = 0  # Index for device selection
    mesh, xfmr_weights = weight_load_helper(LLAMA_1B_PARAMS, ckpt_path, i)

def test_load_weights_multi_device():
    # Load weights in a multi-device setup
    ckpt_path = Path("weights/3B-Instruct")  
    i = 0  # Index for device selection

    # Create mesh if more than one device is used per model
    mesh, xfmr_weights = weight_load_helper(LLAMA_3B_PARAMS, ckpt_path, i)
    print(mesh.devices, mesh.axis_names, mesh.shape)  # Print mesh details for debugging

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
    xfmr_weights = test_load_weights_single_device()  # Load weights for single device
    tokenizer_path: str = "entropix/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)
    engine = EntropixEngine(LLAMA_1B_PARAMS, xfmr_weights, setup_single_device_mesh, tokenizer, xfmr_fn=jax.jit(xfmr, static_argnames=("LLAMA_1B_PARAMS",)),
          sample_fn=jax.jit(sample),)
    decode_state = {
        "next_pos": 0,
        "tokens": jax.numpy.array([[0]]),  # Initialize with appropriate shape and data
        "generated_tokens": jax.numpy.zeros((1, 1), dtype=jax.numpy.int32)
    }
    result = engine.generate(LLAMA_1B_PARAMS, decode_state=decode_state, rng=jax.random.PRNGKey(0))
    assert result is not None

def test_inference_multi_device(setup_multi_device_mesh):
    # Initialize engine with single device
    xfmr_weights = test_load_weights_single_device()  # Load weights for single device
    tokenizer_path: str = "entropix/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)
    engine = EntropixEngine(LLAMA_3B_PARAMS, xfmr_weights, setup_single_device_mesh, tokenizer, xfmr_fn=jax.jit(xfmr, static_argnames=("LLAMA_3B_PARAMS",)),
          sample_fn=jax.jit(sample),)
    decode_state = {
        "next_pos": 0,
        "tokens": jax.numpy.array([[0]]),  # Initialize with appropriate shape and data
        "generated_tokens": jax.numpy.zeros((1, 1), dtype=jax.numpy.int32)
    }
    result = engine.generate(LLAMA_3B_PARAMS, decode_state=decode_state, rng=jax.random.PRNGKey(0))
    assert result is not None