"""JAX glue for tvm-ffi kernels.

Provides :func:`wrap_tvm_ffi_for_jax`, which wraps a compiled tvm-ffi function
so it can be called with JAX arrays inside ``jax.jit`` and ``jax.vmap`` with
zero host-device overhead.

This module requires the ``jax-tvm-ffi`` package
(https://github.com/NVIDIA/jax-tvm-ffi).  Install it with::

    pip install jax-tvm-ffi

How it works
------------
The kernel is registered as an XLA custom call via
``jax_tvm_ffi.register_ffi_target`` and invoked through ``jax.ffi.ffi_call``.
The kernel runs entirely inside XLA's execution pipeline — no
host-device synchronisation, no data copies, zero overhead on GPU.

Limitations
-----------
``jax.grad`` is not supported.  tvm-ffi kernels are opaque to JAX's autodiff.
Use ``jax.custom_vjp`` if gradients are required.

``jax.vmap`` is supported via ``vmap_method="sequential"``, meaning the kernel
is invoked once per batch element rather than as a single batched dispatch.
For batched efficiency, provide a kernel that natively handles a batch dimension.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    import jax

ShapeDtypeSpecifier = Union[
    "jax.ShapeDtypeStruct",
    tuple["jax.ShapeDtypeStruct", ...],
]


def wrap_tvm_ffi_for_jax(
    tvm_fn,
    result_shape_dtypes: Union[ShapeDtypeSpecifier, Callable[..., ShapeDtypeSpecifier]],
    *,
    name: str | None = None,
    platform: str = "cpu",
    arg_spec: list[str] | None = None,
) -> Callable:
    """Wrap a tvm-ffi function so it can be called with JAX arrays.

    The returned callable accepts JAX arrays, is compatible with ``jax.jit``
    and ``jax.vmap``, and runs with zero host-device overhead on GPU.

    Requires the ``jax-tvm-ffi`` package (``pip install jax-tvm-ffi``).

    Parameters
    ----------
    tvm_fn:
        A ``tvm_ffi.Function`` loaded from a compiled tvm-ffi module.
        Must follow the tvm-ffi convention ``fn(out, x, ...)`` — output
        buffer(s) first, then inputs.
    result_shape_dtypes:
        Either a :class:`jax.ShapeDtypeStruct` (or tuple of them) describing
        the output, or a callable that receives the JAX input arrays and
        returns the appropriate struct(s).  Pass a callable when the output
        shape depends on the inputs (e.g. a matrix multiply).
    name:
        Unique name used to register the kernel in JAX's FFI registry.
        Auto-generated if not provided.
    platform:
        ``"cpu"`` or ``"gpu"``.
    arg_spec:
        Argument mapping for ``jax_tvm_ffi.register_ffi_target``.
        Defaults to ``["rets", "args"]``, which matches the tvm-ffi convention
        of output buffer(s) first.

    Returns
    -------
    Callable
        ``(*jax_arrays) -> jax.Array`` (or tuple of arrays for multiple
        outputs), compatible with ``jax.jit`` and ``jax.vmap``.

    Examples
    --------
    .. code-block:: python

        import jax
        import jax.numpy as jnp
        from kernels.jax_tvm_ffi import wrap_tvm_ffi_for_jax

        # kernel = get_kernel("org/my-tvm-ffi-kernel")
        # ops.relu_cuda follows the convention relu_cuda(out, x)

        relu_jax = wrap_tvm_ffi_for_jax(
            ops.relu_cuda,
            result_shape_dtypes=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
            platform="gpu",
        )

        x = jnp.array([-1.0, 0.0, 2.0])
        print(jax.jit(relu_jax)(x))        # runs fully on GPU, zero overhead
        print(jax.vmap(relu_jax)(x[None])) # vmap works, but runs sequentially (N kernel calls)
    """
    import jax
    import jax_tvm_ffi

    target_name = name or f"kernels.tvm_ffi.{uuid.uuid4().hex}"
    arg_spec = arg_spec or ["rets", "args"]

    jax_tvm_ffi.register_ffi_target(
        target_name,
        tvm_fn,
        arg_spec=arg_spec,
        platform=platform,
    )

    if callable(result_shape_dtypes):

        def wrapper(*args):
            return jax.ffi.ffi_call(
                target_name,
                result_shape_dtypes(*args),
                vmap_method="sequential",
            )(*args)

    else:
        _ffi_call = jax.ffi.ffi_call(
            target_name,
            result_shape_dtypes,
            vmap_method="sequential",
        )

        def wrapper(*args):
            return _ffi_call(*args)

    return wrapper
