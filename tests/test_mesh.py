import pytest
import jax
import jax.numpy as jnp
from engine_main import EntropixEngine, create_llama_params, ModelParams
from weights import load_weights, create_partition_spec
from pathlib import Path

# Sample model parameters for 1B and 70B configurations
MODEL_PARAMS = {
    "1B": ModelParams(dim=2048, n_layers=16, n_heads=32, n_kv_heads=8, vocab_size=128256,
                      ffn_dim_multiplier=1.5, multiple_of=256, norm_eps=1e-5,
                      rope_theta=500000.0, use_scaled_rope=True, max_seq_len=4096, num_devices=1),
    "3B": ModelParams(dim=3072, n_layers=28, n_heads=24, n_kv_heads=8, vocab_size=128256,
                       ffn_dim_multiplier=1.5, multiple_of=256, norm_eps=1e-5,
                       rope_theta=500000.0, use_scaled_rope=True, max_seq_len=4096, num_devices=4),
}

@pytest.fixture
def setup_tpu_mesh():
    """Fixture to create a TPU mesh for testing."""
    def create_mesh(num_devices):
        if num_devices == 1:
            return None
        else:
            mesh_shape = jax.experimental.mesh_utils.create_device_mesh((num_devices, 1))
            return jax.sharding.Mesh(mesh_shape, ("dp", "mp"))
    return create_mesh

@pytest.mark.parametrize("load_or_mock_weights", ["weights/1B-Instruct"], indirect=True)
def test_engine_initialization_single_device(setup_tpu_mesh, load_or_mock_weights):
    """Test engine initialization with a single device."""
    params = create_llama_params(MODEL_PARAMS["1B"]._asdict())
    mesh = setup_tpu_mesh(params.num_devices)
    weights, _ = load_or_mock_weights(params)
    engine = EntropixEngine(params, weights, mesh, None, None, None)

    assert engine.mesh is None, "Expected no mesh for single device model."
    assert engine.params.num_devices == 1, "Expected single-device configuration."

@pytest.mark.parametrize("load_or_mock_weights", ["weights/3B-Instruct"], indirect=True)
def test_engine_initialization_multi_device(setup_tpu_mesh, load_or_mock_weights):
    """Test engine initialization with multiple devices."""
    params = create_llama_params(MODEL_PARAMS["3B"]._asdict())
    mesh = setup_tpu_mesh(params.num_devices)
    weights, mesh_used = load_or_mock_weights(params)
    engine = EntropixEngine(params, weights, mesh, None, None, None)

    assert engine.mesh == mesh_used, "Mesh mismatch between weights and engine."
    assert engine.params.num_devices == 8, "Expected 8-device configuration."

@pytest.mark.parametrize("load_or_mock_weights", ["/path/to/real/weights"], indirect=True)
def test_inference_step_single_device(setup_tpu_mesh, load_or_mock_weights):
    """Run inference step with single device to check for errors."""
    params = create_llama_params(MODEL_PARAMS["1B"]._asdict())
    mesh = setup_tpu_mesh(params.num_devices)
    weights, _ = load_or_mock_weights(params)
    engine = EntropixEngine(params, weights, mesh, None, None, None)

    # Simulate a prefill or generate call
    padded_tokens = jnp.zeros((1, 512), dtype=jnp.int32)
    result, _ = engine.prefill(params=params, padded_tokens=padded_tokens, true_length=512)

    assert result is not None, "Expected result from single device inference."
    assert result["tokens"].shape == (6, 1), "Unexpected shape for generated tokens."

@pytest.mark.parametrize("load_or_mock_weights", ["/path/to/real/weights"], indirect=True)
def test_inference_step_multi_device(setup_tpu_mesh, load_or_mock_weights):
    """Run inference step with multiple devices to check for errors in multi-device setup."""
    params = create_llama_params(MODEL_PARAMS["70B"]._asdict())
    mesh = setup_tpu_mesh(params.num_devices)
    weights, _ = load_or_mock_weights(params)
    engine = EntropixEngine(params, weights, mesh, None, None, None)

    # Simulate a prefill or generate call
    padded_tokens = jnp.zeros((8, 512), dtype=jnp.int32)
    result, _ = engine.prefill(params=params, padded_tokens=padded_tokens, true_length=512)

    assert result is not None, "Expected result from multi-device inference."
    assert result["tokens"].shape == (6, 1), "Unexpected shape for generated tokens in multi-device inference."

@pytest.mark.parametrize("model_size, load_or_mock_weights", [("1B", "/path/to/real/weights"), ("70B", "/path/to/real/weights")], indirect=["load_or_mock_weights"])
def test_switch_model_configuration(setup_tpu_mesh, load_or_mock_weights, model_size):
    """Test that switching between model configurations works without errors."""
    params = create_llama_params(MODEL_PARAMS[model_size]._asdict())
    mesh = setup_tpu_mesh(params.num_devices)
    weights, _ = load_or_mock_weights(params)
    engine = EntropixEngine(params, weights, mesh, None, None, None)

    # Simulate a simple inference call to confirm the configuration works
    padded_tokens = jnp.zeros((params.num_devices, 512), dtype=jnp.int32)
    result, _ = engine.prefill(params=params, padded_tokens=padded_tokens, true_length=512)

    assert result is not None, f"Expected result from inference with model {model_size}."
    assert result["tokens"].shape == (6, 1), f"Unexpected shape for tokens with model {model_size}."