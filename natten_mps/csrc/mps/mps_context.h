#pragma once

#include <Metal/Metal.hpp>
#include "mps_buffer_pool.h"

#include <map>
#include <mutex>
#include <string>

namespace natten {
namespace mps {

struct MetalKernel {
    NS::SharedPtr<MTL::Function> function;
    NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

class MetalContext {
public:
    static MetalContext& getInstance();
    NS::SharedPtr<MTL::Device> device;
    NS::SharedPtr<MTL::CommandQueue> commandQueue;
    
    BufferPool& getBufferPool();

    MetalKernel getKernel(
        const std::string& file_name,
        const std::string& function_name,
        const char* kernel_path_cstr,
        const char* extensions_dir_cstr);

private:
    MetalContext();
    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    std::map<std::string, MetalKernel> _kernels;
    std::mutex _mtx;
    std::unique_ptr<BufferPool> _buffer_pool;
};

} // namespace mps
} // namespace natten
