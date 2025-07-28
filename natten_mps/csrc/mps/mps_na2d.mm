#include "natten/mps_na2d.h"
#include "kernels/mps_kernel_structs.h"
#include <ATen/mps/MPSStream.h>
#import <Metal/Metal.h>

extern std::string metallib_path;

// Helper to get the underlying MTLBuffer from a tensor.
static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

namespace natten {
namespace mps {

torch::Tensor na2d_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &value,
    const torch::Tensor &rpb,
    const int64_t kernel_size,
    const int64_t dilation,
    const int64_t is_causal,
    const int64_t original_height,
    const int64_t original_width) {

    @autoreleasepool {
        // Common setup
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(device, "Failed to create MTLDevice.");

        __block NSError *error = nil;
        NSString* metallib_path_ns = [NSString stringWithUTF8String:metallib_path.c_str()];
        NSURL *libraryURL = [NSURL fileURLWithPath:metallib_path_ns];
        id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
        TORCH_CHECK(library, "Failed to load .metallib. Error: ", error.localizedDescription.UTF8String);

        // Options for new tensors
        auto tensor_options = torch::TensorOptions().device(query.device()).dtype(query.dtype());

        // Intermediate attention tensor
        auto attn_sizes = {query.size(0), query.size(1), query.size(2), query.size(3), kernel_size * kernel_size};
        auto attn = torch::empty(attn_sizes, tensor_options);

        // Intermediate context tensor (same shape as value)
        auto context = torch::empty_like(value);

        // Final output tensor with the permuted and reshaped dimensions
        const int64_t batch_size = query.size(0);
        const int64_t all_head_size = value.size(1) * value.size(4);
        auto output_sizes = {batch_size, original_height, original_width, all_head_size};
        auto output = torch::empty(output_sizes, tensor_options);

        // Get command buffer and encoder once
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^{
            id<MTLComputeCommandEncoder> encoder = at::mps::getCurrentMPSStream()->commandEncoder();

            // QK+RPB dispatch
            {
                id<MTLFunction> kernelFunction = [library newFunctionWithName:@"neighborhood_assembly_2d_qkrpb"];
                TORCH_CHECK(kernelFunction, "Failed to find kernel function: neighborhood_assembly_2d_qkrpb");
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
                TORCH_CHECK(pipeline, "Failed to create pipeline state for QK+RPB 2D. Error: ", error.localizedDescription.UTF8String);
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:getMTLBufferStorage(query) offset:query.storage_offset() * query.element_size() atIndex:0];
                [encoder setBuffer:getMTLBufferStorage(key) offset:key.storage_offset() * key.element_size() atIndex:1];
                [encoder setBuffer:getMTLBufferStorage(rpb) offset:rpb.storage_offset() * rpb.element_size() atIndex:2];
                [encoder setBuffer:getMTLBufferStorage(attn) offset:attn.storage_offset() * attn.element_size() atIndex:3];
                NA2dQkrpbProperties props;
                props.batch_size = query.size(0);
                props.heads = query.size(1);
                props.height = query.size(2);
                props.width = query.size(3);
                props.dim = query.size(4);
                props.kernel_size = kernel_size;
                props.dilation = dilation;
                props.is_causal = is_causal;
                props.query_stride_b = query.stride(0);
                props.query_stride_h = query.stride(1);
                props.query_stride_y = query.stride(2);
                props.query_stride_x = query.stride(3);
                props.query_stride_d = query.stride(4);
                props.key_stride_b = key.stride(0);
                props.key_stride_h = key.stride(1);
                props.key_stride_y = key.stride(2);
                props.key_stride_x = key.stride(3);
                props.key_stride_d = key.stride(4);
                props.rpb_stride_h = rpb.stride(0);
                props.rpb_stride_y = rpb.stride(1);
                props.rpb_stride_x = rpb.stride(2);
                props.attn_stride_b = attn.stride(0);
                props.attn_stride_h = attn.stride(1);
                props.attn_stride_y = attn.stride(2);
                props.attn_stride_x = attn.stride(3);
                props.attn_stride_k = attn.stride(4);
                [encoder setBytes:&props length:sizeof(props) atIndex:4];
                MTLSize gridSize = MTLSizeMake(props.width, props.height, props.batch_size * props.heads);
                MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            }

            // AV dispatch
            {
                id<MTLFunction> kernelFunction = [library newFunctionWithName:@"neighborhood_assembly_2d_av"];
                TORCH_CHECK(kernelFunction, "Failed to find kernel function: neighborhood_assembly_2d_av");
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
                TORCH_CHECK(pipeline, "Failed to create pipeline state for AV 2D. Error: ", error.localizedDescription.UTF8String);
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:getMTLBufferStorage(attn) offset:attn.storage_offset() * attn.element_size() atIndex:0];
                [encoder setBuffer:getMTLBufferStorage(value) offset:value.storage_offset() * value.element_size() atIndex:1];
                [encoder setBuffer:getMTLBufferStorage(context) offset:context.storage_offset() * context.element_size() atIndex:2];
                NA2dAVProperties props;
                props.batch_size = attn.size(0);
                props.heads = attn.size(1);
                props.height = attn.size(2);
                props.width = attn.size(3);
                props.dim = value.size(4);
                props.kernel_size = kernel_size;
                props.dilation = dilation;
                props.is_causal = is_causal;
                props.attn_stride_b = attn.stride(0);
                props.attn_stride_h = attn.stride(1);
                props.attn_stride_y = attn.stride(2);
                props.attn_stride_x = attn.stride(3);
                props.attn_stride_k = attn.stride(4);
                props.value_stride_b = value.stride(0);
                props.value_stride_h = value.stride(1);
                props.value_stride_y = value.stride(2);
                props.value_stride_x = value.stride(3);
                props.value_stride_d = value.stride(4);
                props.output_stride_b = context.stride(0);
                props.output_stride_h = context.stride(1);
                props.output_stride_y = context.stride(2);
                props.output_stride_x = context.stride(3);
                props.output_stride_d = context.stride(4);
                [encoder setBytes:&props length:sizeof(props) atIndex:3];
                MTLSize gridSize = MTLSizeMake(props.width, props.height, props.batch_size * props.heads);
                MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            }

            // Permute and Reshape dispatch
            {
                id<MTLFunction> kernelFunction = [library newFunctionWithName:@"na2d_permute_and_reshape"];
                TORCH_CHECK(kernelFunction, "Failed to find kernel function: na2d_permute_and_reshape");
                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
                TORCH_CHECK(pipeline, "Failed to create pipeline state for Permute/Reshape 2D. Error: ", error.localizedDescription.UTF8String);
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:getMTLBufferStorage(context) offset:context.storage_offset() * context.element_size() atIndex:0];
                [encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:1];
                PermuteAndReshape2dProperties props;
                props.batch_size = context.size(0);
                props.heads = context.size(1);
                props.height = original_height;
                props.width = original_width;
                props.dim = context.size(4);
                props.context_stride_b = context.stride(0);
                props.context_stride_h = context.stride(1);
                props.context_stride_y = context.stride(2);
                props.context_stride_x = context.stride(3);
                props.context_stride_d = context.stride(4);
                props.output_stride_b = output.stride(0);
                props.output_stride_y = output.stride(1);
                props.output_stride_x = output.stride(2);
                props.output_stride_d = output.stride(3);
                [encoder setBytes:&props length:sizeof(props) atIndex:2];
                MTLSize gridSize = MTLSizeMake(props.width, props.height, props.batch_size);
                MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            }
        });
        return output;
    }
}

} // namespace mps
} // namespace natten