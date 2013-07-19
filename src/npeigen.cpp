#include "npeigen.hpp"

static py::object numpy;
py::object GetNumPyMod() {
  if (numpy.is_none()) {
    numpy = py::import("numpy");
  }
  return numpy;
}

template<int IntSize> struct IntName { static const char* npname; };
template<> const char* IntName<4>::npname = "int32";
template<> const char* IntName<8>::npname = "int64";
template<> const char* NPMatrixTypes<int>::scalar_npname = IntName<sizeof(int)>::npname;

template<> const char* NPMatrixTypes<float>::scalar_npname = "float32";
template<> const char* NPMatrixTypes<double>::scalar_npname = "float64";
