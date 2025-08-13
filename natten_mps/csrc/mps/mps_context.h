#pragma once

#include <map>
#include <mutex>
#include <string>

// Forward-declare Objective-C types to avoid importing Metal in the header.
#ifdef __OBJC__
@protocol MTLDevice, MTLCommandQueue, MTLFunction, MTLComputePipelineState, MTLLibrary;
#else
typedef void* id;
#endif

namespace natten {
namespace mps {

struct MetalKernel {
    id<MTLFunction> function;
    id<MTLComputePipelineState> pipeline;
};

class MetalContext {
public:
    static MetalContext& getInstance();
    
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    
    MetalKernel getKernel(const std::string& function_name);

private:
    MetalContext();
    ~MetalContext();
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    std::map<std::string, MetalKernel> _kernels;
    id<MTLLibrary> _library;
    std::mutex _mtx;
};

} // namespace mps
} // namespace natten