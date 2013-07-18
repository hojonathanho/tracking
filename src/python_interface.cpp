#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "mass_system.hpp"

namespace py = boost::python;

// From https://raw.github.com/mapnik/pymapnik2/master/cpp/python_optional.hpp
template <typename T, typename X1 = py::detail::not_specified, typename X2 = py::detail::not_specified, typename X3 = py::detail::not_specified>
struct class_with_converter : public py::class_<T, X1, X2, X3> {
  typedef class_with_converter<T,X1,X2,X3> self;
  // Construct with the class name, with or without docstring, and default __init__() function
  class_with_converter(char const* name, char const* doc = 0) : py::class_<T, X1, X2, X3>(name, doc)  { }

  // Construct with class name, no docstring, and an uncallable __init__ function
  class_with_converter(char const* name, py::no_init_t y) : py::class_<T, X1, X2, X3>(name, y) { }

  // Construct with class name, docstring, and an uncallable __init__ function
  class_with_converter(char const* name, char const* doc, py::no_init_t y) : py::class_<T, X1, X2, X3>(name, doc, y) { }

  // Construct with class name and init<> function
  template <class DerivedT> class_with_converter(char const* name, py::init_base<DerivedT> const& i)
      : py::class_<T, X1, X2, X3>(name, i) { }

  // Construct with class name, docstring and init<> function
  template <class DerivedT>
  inline class_with_converter(char const* name, char const* doc, py::init_base<DerivedT> const& i)
      : py::class_<T, X1, X2, X3>(name, doc, i) { }

  template <class D>
  self& def_readwrite_convert(char const* name, D const& d, char const* doc=0) {
    this->add_property(
      name,
      py::make_getter(d, py::return_value_policy<py::return_by_value>()),
      py::make_setter(d, py::default_call_policies())
    );
    return *this;
  }
};


BOOST_PYTHON_MODULE(trackingpy) {
  py::to_python_converter<NPMatrixd, NPMatrixd::ToPythonConverter>();
  NPMatrixd::FromPythonConverter();

  // Converters for Vector3d
  py::to_python_converter<Eigen::Vector3d, EigenMatrixConverters<double, 3, 1, 0, 3, 1>::ToPython>();
  EigenMatrixConverters<double, 3, 1, 0, 3, 1>::FromPython();

  using tracking::MassSystem;

  class_with_converter<MassSystem::SimulationParams, boost::noncopyable>("SimulationParams")
    .def_readwrite_convert("gravity", &MassSystem::SimulationParams::gravity)
    .def_readwrite("dt", &MassSystem::SimulationParams::dt)
    .def_readwrite("solver_iters", &MassSystem::SimulationParams::solver_iters)
    ;

  py::class_<MassSystem>("MassSystem", py::init<const NPMatrixd&, const NPMatrixd&, const MassSystem::SimulationParams&>())
    .def("step", &MassSystem::step)
    .def("get_node_positions", &MassSystem::get_node_positions)
    .def("add_anchor_constraint", &MassSystem::add_anchor_constraint)
    .def("add_distance_constraint", &MassSystem::add_distance_constraint)
    .def("enable_constraint", &MassSystem::enable_constraint)
    .def("disable_constraint", &MassSystem::disable_constraint)
    ;
}
