#pragma once

#include "npeigen.hpp"
#include <boost/python.hpp>

namespace tracking {

namespace py = boost::python;

class MassSystem {
public:

  // struct MassSystemParams {
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
    Eigen::Vector3d gravity;
    double damping;
  };

  MassSystem(const NPMatrixd& init_x, const NPMatrixd& m, const SimulationParams& sim_params);
  ~MassSystem();

  int add_anchor_constraint(int i_point, const NPMatrixd& anchor_pos);
  int add_distance_constraint(int i_point1, int i_point2, double resting_len);
  int add_plane_constraint(int i_point, const Eigen::Vector3d& plane_point, const Eigen::Vector3d& plane_normal);
  void disable_constraint(int i);
  void enable_constraint(int i);
  void randomize_constraints();

  void step();

  py::object get_node_positions() const;


private:
  class Impl;
  Impl *m_impl;
};

} // namespace tracking
