from .. import relu

try:
    import torch
    from torch import nn

    class ReLU(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            relu(x, out)
            return out

except ImportError:
    pass

try:
    import jax

    from kernels.jax_tvm_ffi import wrap_tvm_ffi_for_jax

    from .._ops import ops as _ops

    _relu_jax_fns = {}

    def relu_jax(x: "jax.Array") -> "jax.Array":
        platform = next(iter(x.devices())).platform
        if platform not in _relu_jax_fns:
            if platform == "cpu":
                tvm_fn = _ops.relu_cpu
            elif platform == "gpu":
                tvm_fn = _ops.relu_cuda
            else:
                raise NotImplementedError(f"JAX relu not supported on platform: {platform}")
            _relu_jax_fns[platform] = wrap_tvm_ffi_for_jax(
                tvm_fn, lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), platform=platform
            )
        return _relu_jax_fns[platform](x)

except ImportError:
    pass
