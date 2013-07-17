#pragma once

#include "npeigen.hpp"
#include <boost/python.hpp>

namespace tracking {

namespace py = boost::python;

class Cloth {
public:

  // struct ClothParams {
  //   NPMatrixd init_x, init_v, m;

  //   // boost python compatibility
  //   void set_init_x(const boost::python::object& o) { init_x = o; }
  //   boost::python::object get_init_x() { return init_x.ndarray(); }

  //   void set_init_v(const boost::python::object& o) { init_v = o; }
  //   boost::python::object get_init_v() { return init_v.ndarray(); }

  //   void set_m(const boost::python::object& o) { m = o; }
  //   boost::python::object get_m() { return m.ndarray(); }
  // };

  struct SimulationParams {
    double dt;
    int solver_iters;
    NPMatrixd gravity;

    // SimulationParams() {
    //   dt = .01;
    //   solver_iters = 2;
    //   gravity = Eigen::Vector3d(0, 0, -20);
    //   std::cout << "plain old constructor " << gravity << std::endl;
    // }
    // SimulationParams(const SimulationParams& other) {
    //   std::cout << "explicit copy constructor" << std::endl;
    //   dt = other.dt;
    //   solver_iters = other.solver_iters;
    //   gravity = other.gravity;
    // }
  };

  Cloth(const NPMatrixd& init_x, const NPMatrixd& m, const SimulationParams& sim_params);
  ~Cloth();

  void step();
  py::object getNodePositions() const;


private:
  class Impl;
  Impl *m_impl;
};

} // namespace tracking
