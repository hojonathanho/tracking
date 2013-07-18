#include "npeigen.hpp"

static py::object numpy;
py::object GetNumPyMod() {
  if (numpy.is_none()) {
    numpy = py::import("numpy");
  }
  return numpy;
}

#if __x86_64__
  template<> const char* NPMatrixTypes<int>::scalar_npname = "int64";
#else
  template<> const char* NPMatrixTypes<int>::scalar_npname = "int32";
#endif
template<> const char* NPMatrixTypes<float>::scalar_npname = "float32";
template<> const char* NPMatrixTypes<double>::scalar_npname = "float64";
