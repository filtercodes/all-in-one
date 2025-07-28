#!/usr/bin/env python
import glob
import os
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "13.0")
os.environ["CXX"] = "clang++"

class MetalBuildExtension(BuildExtension):
    def run(self):
        print("--- Compiling Metal shaders into a .metallib library ---")

        project_name = "natten"
        metal_kernels_dir = os.path.join(project_name, "mps", "kernels")
        output_library_path = os.path.join(metal_kernels_dir, "natten.metallib")
        
        os.makedirs(metal_kernels_dir, exist_ok=True)

        # Find all .metal source files.
        metal_files = glob.glob(os.path.join("csrc", "mps", "kernels", "*.metal"))
        if not metal_files:
            raise RuntimeError(f"No .metal files found in 'csrc/mps/kernels'. Cannot build Metal library.")

        print(f"Found Metal source files: {metal_files}")

        # Define the path to the shared kernel helpers header
        structs_header_dir = os.path.join("csrc", "include")

        # compile all .metal files into a single .metallib.
        try:
            command = [
                "xcrun", "-sdk", "macosx", "metal",
            ] + metal_files + [
                "-o", output_library_path
            ]
            print(f"Executing Metal compiler command: {' '.join(command)}")
            subprocess.check_call(command)
            print(f"Successfully compiled Metal library to: {output_library_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to compile Metal files: {e}") from e
        except FileNotFoundError:
            raise RuntimeError("Failed to run `xcrun`. Is Xcode Command Line Tools installed?") from None

        super().run()

# =================================================================================
# Main Setup Configuration
# =================================================================================

extensions_dir = os.path.abspath("csrc")

# Find all C++ and Objective-C++ source files for the extension.
sources = glob.glob(os.path.join(extensions_dir, "*.mm")) + \
          glob.glob(os.path.join(extensions_dir, "mps", "*.mm"))

print(f"Found C++/MM source files: {sources}")

ext_modules = [
    CppExtension(
        "natten._C",
        sources,
        include_dirs=[os.path.join(extensions_dir, "include")],
        extra_compile_args=[
            "-std=c++17",
            "-O3",
            "-g",
            "-mmacosx-version-min=13.0",
            "-framework", "Metal",
            "-framework", "Foundation",
            "-framework", "MetalPerformanceShaders",
        ],
        extra_link_args=[
            "-framework", "Metal",
            "-framework", "Foundation",
            "-framework", "MetalPerformanceShaders",
        ],
    )
]

setup(
    name="natten-mps",
    version="0.1.1",
    author="Stefan Miletic",
    description="Neighborhood Attention Extension for PyTorch (MPS Backend)",
    packages=find_packages(),
    # Package the compiled .metallib instead of the source .metal files.
    package_data={
        "natten.mps.kernels": ["natten.metallib"],
    },
    include_package_data=True,
    ext_modules=ext_modules,
    # Use our custom build class.
    cmdclass={"build_ext": MetalBuildExtension},
    zip_safe=False,
)
