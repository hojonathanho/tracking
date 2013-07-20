#pragma once

#include "npeigen.hpp"
#include <boost/python.hpp>

namespace tracking {

namespace py = boost::python;

class MassSystem {
public:

  struct SimulationParams {
    double dt;
    int solver_iters;
    Eigen::Vector3d gravity;
    double damping;

    double stretching_stiffness;
    double bending_stiffness;
  };

  MassSystem(const NPMatrixd& init_x, const NPMatrixd& m, const SimulationParams& sim_params);
  ~MassSystem();

  // Takes an integer (num_triangles x 3) array that specifies the indices of triangle vertices
  // and populates the triangle tree according to the current position values
  void declare_triangles(const NPMatrixi& triangles);
  int triangle_ray_test(const Eigen::Vector3d &ray_from, const Eigen::Vector3d &ray_to) const;
  std::vector<int> triangle_ray_test_against_nodes(const Eigen::Vector3d &ray_from) const;

  // Adds constraints and returns constraint ids
  int add_anchor_constraint(int i_point, const NPMatrixd& anchor_pos);
  int add_plane_constraint(int i_point, const Eigen::Vector3d& plane_point, const Eigen::Vector3d& plane_normal);
  int add_distance_constraint(int i_point1, int i_point2, double resting_len);
  int add_bending_constraint(int i1, int i2, int i3, int i4, double resting_angle);
  // Enable or disable constraints. Pass in the id returned by the add_ method.
  void disable_constraint(int i);
  void enable_constraint(int i);
  // Call this after adding all constraints to randomize solving order
  void randomize_constraints();


  void apply_forces(const NPMatrixd& f);
  // Steps the simulation
  void step();

  py::object get_node_positions() const;


private:
  struct Impl;
  Impl *m_impl;
};

} // namespace tracking
