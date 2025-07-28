import torch
from torch.autograd import Function
import os
from . import _C

# --- MPS Backend Initialization ---
def _init_mps_backend():
    """
    Called automatically when this module is imported.
    """
    # Use the location of the compiled extension to find the package path
    package_path = os.path.dirname(os.path.abspath(_C.__file__))
    metallib_path = os.path.join(package_path, "mps", "kernels", "natten.metallib")
    if not os.path.exists(metallib_path):
        raise RuntimeError(
            f"NATTEN Metal library not found at '{metallib_path}'. "
            "Please reinstall the package to ensure it's compiled correctly."
        )
    _C.init_natten_mps(metallib_path)
    return True

_mps_backend_initialized = _init_mps_backend()

class NATTEN1DFunction(Function):
    @staticmethod
    def forward(ctx, query, key, value, rpb, kernel_size, dilation, original_length):
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        is_causal = False # TODO: make is_causal a real parameter
        output = _C.na1d_forward(
            query, 
            key,
            value,
            rpb,
            kernel_size, 
            dilation, 
            is_causal,
            original_length)
        ctx.save_for_backward(query, key, value)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

class NATTEN2DFunction(Function):
    @staticmethod
    def forward(ctx, query, key, value, rpb, kernel_size, dilation, original_height, original_width):
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        is_causal = False # TODO: make is_causal a real parameter
        output = _C.na2d_forward(
            query, 
            key,
            value,
            rpb,
            kernel_size, 
            dilation, 
            is_causal,
            original_height,
            original_width)
        ctx.save_for_backward(query, key, value)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

def natten1d(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, rpb: torch.Tensor, kernel_size: int, dilation: int, original_length: int):
    return NATTEN1DFunction.apply(query, key, value, rpb, kernel_size, dilation, original_length)

def natten2d(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, rpb: torch.Tensor, kernel_size: int, dilation: int, original_height: int, original_width: int):
    return NATTEN2DFunction.apply(query, key, value, rpb, kernel_size, dilation, original_height, original_width)
