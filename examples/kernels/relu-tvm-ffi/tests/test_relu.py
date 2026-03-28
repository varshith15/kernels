import numpy as np
import pytest
import torch
import torch.nn.functional as F

import relu_tvm_ffi


@pytest.mark.kernels_ci
def test_relu(device):
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    torch.testing.assert_close(F.relu(x), relu_tvm_ffi.relu(x, torch.empty_like(x)))


@pytest.mark.kernels_ci
def test_relu_views(device):
    x = torch.arange(-20, 20, device=device, dtype=torch.float32)

    # Keep buffers and fill on each iteration. Stable pointers make C++-side
    # pointer inspection esier.
    out = torch.empty_like(x)
    out_check = torch.empty_like(x)

    for i in range(41):
        # Put a sentineal value in the output.
        out.fill_(42)
        out_check.fill_(42)

        relu_tvm_ffi.relu(x[i:], out[i:])
        out_check[i:] = F.relu(x[i:])

        torch.testing.assert_close(out, out_check)


@pytest.mark.kernels_ci
def test_relu_layer(device):
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    layer = relu_tvm_ffi.layers.ReLU()
    torch.testing.assert_close(F.relu(x), layer(x))


@pytest.mark.kernels_ci
def test_relu_jax(device):
    jax = pytest.importorskip("jax")
    pytest.importorskip("jax_tvm_ffi", reason="jax-tvm-ffi not installed")

    import jax.numpy as jnp

    if device.type == "cpu":
        jax_device = jax.devices("cpu")[0]
    elif device.type == "cuda":
        try:
            jax_device = jax.devices("gpu")[0]
        except RuntimeError:
            pytest.skip("JAX GPU device not available")
    else:
        pytest.skip(f"JAX test not supported for device: {device.type}")

    x_np = np.random.randn(1024).astype(np.float32)
    x = jax.device_put(jnp.array(x_np), jax_device)
    result = relu_tvm_ffi.layers.relu_jax(x)
    np.testing.assert_allclose(np.array(result), np.maximum(0, x_np), rtol=1e-5)
