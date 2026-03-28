import importlib.util
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


has_jax = importlib.util.find_spec("jax") is not None
has_torch = importlib.util.find_spec("torch") is not None
has_tvm_ffi = importlib.util.find_spec("tvm_ffi") is not None


__all__ = ["has_jax", "has_torch", "has_tvm_ffi", "tomllib"]
