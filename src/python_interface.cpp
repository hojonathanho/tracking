#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "cloth.hpp"

#include <iostream>
using namespace std;

namespace py = boost::python;
using tracking::Cloth;

// From https://raw.github.com/mapnik/pymapnik2/master/cpp/python_optional.hpp
template <typename T, typename X1 = boost::python::detail::not_specified, typename X2 = boost::python::detail::not_specified, typename X3 = boost::python::detail::not_specified>
class class_with_converter : public boost::python::class_<T, X1, X2, X3>
{
public:
    typedef class_with_converter<T,X1,X2,X3> self;
    // Construct with the class name, with or without docstring, and default __init__() function
    class_with_converter(char const* name, char const* doc = 0) : boost::python::class_<T, X1, X2, X3>(name, doc)  { }

    // Construct with class name, no docstring, and an uncallable __init__ function
    class_with_converter(char const* name, boost::python::no_init_t y) : boost::python::class_<T, X1, X2, X3>(name, y) { }

    // Construct with class name, docstring, and an uncallable __init__ function
    class_with_converter(char const* name, char const* doc, boost::python::no_init_t y) : boost::python::class_<T, X1, X2, X3>(name, doc, y) { }

    // Construct with class name and init<> function
    template <class DerivedT> class_with_converter(char const* name, boost::python::init_base<DerivedT> const& i)
        : boost::python::class_<T, X1, X2, X3>(name, i) { }

    // Construct with class name, docstring and init<> function
    template <class DerivedT>
    inline class_with_converter(char const* name, char const* doc, boost::python::init_base<DerivedT> const& i)
        : boost::python::class_<T, X1, X2, X3>(name, doc, i) { }

    template <class D>
    self& def_readwrite_convert(char const* name, D const& d, char const* doc=0)
    {
        this->add_property(name,
                           boost::python::make_getter(d, boost::python::return_value_policy<boost::python::return_by_value>()),
                           boost::python::make_setter(d, boost::python::default_call_policies()));
        return *this;
    }
};


BOOST_PYTHON_MODULE(trackingpy) {
  py::to_python_converter<NPMatrixd, NPMatrixd::ToPythonConverter>();
  NPMatrixd::FromPythonConverter();

  // py::class_<Cloth::ClothParams>("ClothParams")
  //   .add_property("init_x", &Cloth::ClothParams::get_init_x, &Cloth::ClothParams::set_init_x)
  //   .add_property("init_v", &Cloth::ClothParams::get_init_v, &Cloth::ClothParams::set_init_v)
  //   .add_property("m", &Cloth::ClothParams::m, &Cloth::ClothParams::m)
  //   ;

  class_with_converter<Cloth::SimulationParams, boost::noncopyable>("SimulationParams")
    .def_readwrite_convert("gravity", &Cloth::SimulationParams::gravity)
    .def_readwrite("dt", &Cloth::SimulationParams::dt)
    .def_readwrite("solver_iters", &Cloth::SimulationParams::solver_iters)
    ;



  // py::class_<NPMatrixd>("NPMatrixd", py::init<const py::object&>());

  py::class_<Cloth>("Cloth", py::init<const NPMatrixd&, const NPMatrixd&, const Cloth::SimulationParams&>())
    .def("step", &Cloth::step)
    .def("get_node_positions", &Cloth::get_node_positions)
    .def("add_anchor_constraint", &Cloth::add_anchor_constraint)
    .def("add_distance_constraint", &Cloth::add_distance_constraint)
    .def("enable_constraint", &Cloth::enable_constraint)
    .def("disable_constraint", &Cloth::disable_constraint)
    ;
}
