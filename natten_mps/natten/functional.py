import torch
from torch.autograd import Function
import os
import sys
import importlib.resources as importlib_resources
from . import _C

# --- MPS Backend Initialization ---
_MPS_BACKEND_INITIALIZED = False

def _init_mps_backend():
    """
    Initializes the MPS backend by locating and loading the Metal library.
    This function is called lazily on the first forward pass.
    """
    global _MPS_BACKEND_INITIALIZED
    if _MPS_BACKEND_INITIALIZED:
        return
    
    # Use the recommended way to find package data
    with importlib_resources.path("natten.mps.kernels", "natten.metallib") as p:
        metallib_path = str(p)
    
    if not os.path.exists(metallib_path):
        raise RuntimeError(
            f"NATTEN Metal library not found at '{metallib_path}'. "
            "Please reinstall the package to ensure it's compiled correctly."
        )
    _C.init_natten_mps(metallib_path)
    _MPS_BACKEND_INITIALIZED = True

class NATTEN1DFunction(Function):
    @staticmethod
    def forward(ctx, query, key, value, rpb, kernel_size, dilation, original_length):
        _init_mps_backend()
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
        _init_mps_backend()
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
