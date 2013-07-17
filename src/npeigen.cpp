#include "npeigen.hpp"

static boost::python::object numpy;
boost::python::object GetNumPyMod() {
  if (numpy.is_none()) {
    numpy = boost::python::import("numpy");
  }
  return numpy;
}

template<> const char* NPMatrixTypes<int>::scalar_npname = "int32";
template<> const char* NPMatrixTypes<float>::scalar_npname = "float32";
template<> const char* NPMatrixTypes<double>::scalar_npname = "float64";
