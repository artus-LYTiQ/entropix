import pytest
import jax
from jax.sharding import Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from pathlib import Path

from entropix.engine import EntropixEngine
from entropix.weights import load_weights, create_partition_spec
from entropix.config import Params, LLAMA_1B_PARAMS, LLAMA_3B_PARAMS

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

def test_load_weights_single_device():
    # Load weights in a single device setup
    ckpt_path: Path = Path("weights/1B-Instruct")  
    i = 0  # Index for device selection
    devices = jax.devices()[i * LLAMA_1B_PARAMS.num_devices : (i + 1) * LLAMA_1B_PARAMS.num_devices]

    # Create mesh if more than one device is used per model
    if LLAMA_1B_PARAMS.num_devices > 1:
        mesh = Mesh(devices, ("x",))
    else:
        mesh = None  # Single-device case
        context = jax.default_device(devices[0])

    # Load weights using the provided mesh
    with context:
        xfmr_weights, _ = load_weights(ckpt_path, LLAMA_1B_PARAMS, mesh)
        assert xfmr_weights is not None
    assert mesh is None  # No mesh for single device

def test_load_weights_multi_device(setup_multi_device_mesh):
    # Load weights in a multi-device setup
    ckpt_dir = Path("weights/3B-Instruct")  

    i = 0  # Index for device selection
    devices = jax.devices()[i * LLAMA_3B_PARAMS.num_devices : (i + 1) * LLAMA_3B_PARAMS.num_devices]

    # Create mesh if more than one device is used per model
    if LLAMA_3B_PARAMS.num_devices > 1:
        mesh = Mesh(devices, ("x",))
        context = mesh
    else:
        mesh = None  # Single-device case
        context = jax.default_device(devices[0])
    
        # Load weights using the provided mesh
    with context:
        xfmr_weights, _ = load_weights(ckpt_dir, LLAMA_1B_PARAMS, mesh)
        assert xfmr_weights is not None

    assert isinstance(mesh, Mesh)  # Mesh should be present
    print(mesh.devices, mesh.axis_names, mesh.shape)  # Print mesh details for debugging

def test_engine_initialization_single_device(setup_single_device_mesh):
    # Initialize engine with single device
    xfmr_weights = ...  # Mock or actual load
    tokenizer = ...  # Mock or actual tokenizer load
    engine = EntropixEngine(LLAMA_1B_PARAMS, xfmr_weights, setup_single_device_mesh, tokenizer, None, None)
    assert engine.mesh == setup_single_device_mesh  # Should match single device (None)

def test_engine_initialization_multi_device(setup_multi_device_mesh):
    # Initialize engine with multi-device
    xfmr_weights = ...  # Mock or actual load
    tokenizer = ...  # Mock or actual tokenizer load
    engine = EntropixEngine(LLAMA_3B_PARAMS, xfmr_weights, setup_multi_device_mesh, tokenizer, None, None)
    assert engine.mesh == setup_multi_device_mesh  # Should match multi-device mesh

def test_inference_single_device(setup_single_device_mesh):
    # Run inference test on single device
    xfmr_weights = ...  # Mock or actual load
    tokenizer = ...  # Mock or actual tokenizer load
    engine = EntropixEngine(LLAMA_1B_PARAMS, xfmr_weights, setup_single_device_mesh, tokenizer, None, None)
    result = engine.generate(LLAMA_1B_PARAMS, decode_state={}, rng=jax.random.PRNGKey(0))
    assert result is not None

def test_inference_multi_device(setup_multi_device_mesh):
    # Run inference test on multi-device setup
    xfmr_weights = ...  # Mock or actual load
    tokenizer = ...  # Mock or actual tokenizer load
    engine = EntropixEngine(LLAMA_3B_PARAMS, xfmr_weights, setup_multi_device_mesh, tokenizer, None, None)
    result = engine.generate(LLAMA_3B_PARAMS, decode_state={}, rng=jax.random.PRNGKey(0))
    assert result is not None