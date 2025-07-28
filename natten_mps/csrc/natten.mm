#include <torch/extension.h>
#include "include/natten/mps_na1d.h"
#include "include/natten/mps_na2d.h"
#include <string>

std::string metallib_path;

void init_natten_mps(const std::string& path) {
    metallib_path = path;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_natten_mps", &init_natten_mps, "Initialize NATTEN MPS Backend");
  m.def("na1d_forward", &natten::mps::na1d_forward, "NATTEN 1D forward (MPS)",
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("rpb"),
        py::arg("kernel_size"), py::arg("dilation"), py::arg("is_causal"),
        py::arg("original_length"));
  m.def("na2d_forward", &natten::mps::na2d_forward, "NATTEN 2D forward (MPS)",
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("rpb"),
        py::arg("kernel_size"), py::arg("dilation"), py::arg("is_causal"),
        py::arg("original_height"), py::arg("original_width"));
}
