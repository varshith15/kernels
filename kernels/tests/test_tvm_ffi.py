import numpy as np
import pytest

from kernels import get_kernel, get_local_kernel, install_kernel

relu_supported_devices = ["cpu", "cuda", "xpu"]


def test_relu_load(device):
    if device not in relu_supported_devices:
        pytest.skip(f"Device is not one of: {','.join(relu_supported_devices)}")
    get_kernel("kernels-test/relu-tvm-ffi", version=1)


@pytest.mark.torch_only
def test_relu_torch(device):
    if device not in relu_supported_devices:
        pytest.skip(f"Device is not one of: {','.join(relu_supported_devices)}")
    kernel = get_kernel("kernels-test/relu-tvm-ffi", version=1)

    import torch

    x = torch.arange(-10, 10, dtype=torch.float32, device=device)
    out = kernel.relu(x, torch.empty_like(x))

    torch.testing.assert_close(out, torch.relu(x))


def test_relu_jax(device):
    if device not in relu_supported_devices:
        pytest.skip(f"Device is not one of: {','.join(relu_supported_devices)}")

    jax = pytest.importorskip("jax")
    pytest.importorskip("jax_tvm_ffi", reason="jax-tvm-ffi not installed")

    import jax.numpy as jnp

    from kernels.jax_tvm_ffi import wrap_tvm_ffi_for_jax

    jax_platform = {"cpu": "cpu", "cuda": "gpu"}.get(device)
    if jax_platform is None:
        pytest.skip(f"JAX does not support device: {device}")

    kernel = get_kernel("kernels-test/relu-tvm-ffi", version=1)
    # Access raw tvm-ffi ops directly — no public API exists for this yet.
    tvm_fn = kernel._ops.ops.relu_cpu if device == "cpu" else kernel._ops.ops.relu_cuda

    jax_devices = [d for d in jax.devices() if d.platform == jax_platform]
    if not jax_devices:
        pytest.skip(f"No JAX {jax_platform} device available")

    relu_jax = jax.jit(wrap_tvm_ffi_for_jax(tvm_fn, lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), platform=jax_platform))

    x = jax.device_put(jnp.arange(-10, 10, dtype=jnp.float32), jax_devices[0])
    np.testing.assert_allclose(np.array(relu_jax(x)), np.maximum(0, np.array(x)))


def test_local_load(device):
    if device not in relu_supported_devices:
        pytest.skip(f"Device is not one of: {','.join(relu_supported_devices)}")

    package_name, path = install_kernel("kernels-test/relu-tvm-ffi", "v1")
    get_local_kernel(path.parent.parent, package_name)
