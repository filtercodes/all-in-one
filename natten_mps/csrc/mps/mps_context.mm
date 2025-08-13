#include "mps_context.h"
#include <iostream>
#import <Metal/Metal.h>
#include <torch/extension.h>

extern std::string metallib_path;

namespace natten {
namespace mps {

MetalContext& MetalContext::getInstance() {
    static MetalContext instance;
    return instance;
}

MetalContext::MetalContext() {
    @autoreleasepool {
        std::cout << "NATTEN MPS: Initializing MetalContext Singleton." << std::endl;
        device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(device, "Failed to create MTLDevice.");
        [device retain];

        commandQueue = [device newCommandQueue];
        TORCH_CHECK(commandQueue, "Failed to create MTLCommandQueue.");
        [commandQueue retain];

        _library = nil;
    }
}

MetalContext::~MetalContext() {
    @autoreleasepool {
        for (auto const& [key, val] : _kernels) {
            [val.function release];
            [val.pipeline release];
        }
        [_library release];
        [commandQueue release];
        [device release];
    }
}

MetalKernel MetalContext::getKernel(const std::string& function_name) {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _kernels.find(function_name);
    if (it != _kernels.end()) {
        return it->second;
    }

    @autoreleasepool {
        if (!_library) {
            TORCH_CHECK(!metallib_path.empty(), "Metal library path not initialized. Did you call init_natten_mps(path)?");
            
            __block NSError *error = nil;
            NSString* metallib_path_ns = [NSString stringWithUTF8String:metallib_path.c_str()];
            NSURL *libraryURL = [NSURL fileURLWithPath:metallib_path_ns];
            
            _library = [device newLibraryWithURL:libraryURL error:&error];
            TORCH_CHECK(_library, "Failed to load .metallib. Error: ", error.localizedDescription.UTF8String);
            [_library retain];
        }

        __block NSError *error = nil;
        NSString* function_name_ns = [NSString stringWithUTF8String:function_name.c_str()];
        id<MTLFunction> function = [_library newFunctionWithName:function_name_ns];
        TORCH_CHECK(function, "Failed to find kernel function: ", function_name);

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        TORCH_CHECK(pipeline, "Failed to create pipeline state for ", function_name, ". Error: ", error.localizedDescription.UTF8String);

        [pipeline retain];

        MetalKernel kernel = { function, pipeline };
        _kernels[function_name] = kernel;
        return kernel;
    }
}

} // namespace mps
} // namespace natten