#include "cloth.hpp"

#include <iostream>
using namespace std;

#include <Eigen/Core>
using namespace Eigen;

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace tracking {

struct PositionConstraint {
  enum Type { ANCHOR=0, DISTANCE };
  int m_id;
  virtual Type getType() const = 0;
  virtual void enforce(NPMatrixd& x) = 0;
};
struct AnchorConstraint : public PositionConstraint {
  int m_i_point; Vector3d m_anchor_pos;
  AnchorConstraint(int i_point, const Vector3d &anchor_pos) : m_i_point(i_point), m_anchor_pos(anchor_pos) { }
  virtual Type getType() const { return ANCHOR; }
  virtual void enforce(NPMatrixd& x) {
    x.row(m_i_point) = m_anchor_pos;
  }
};
struct DistanceConstraint : public PositionConstraint {
  int m_i_point1; int m_i_point2; double m_invm_point1, m_invm_point2; double m_resting_len;
  DistanceConstraint(int i_point1, int i_point2, double invm_point1, double invm_point2, double resting_len) :
    m_i_point1(i_point1), m_i_point2(i_point2), m_invm_point1(invm_point1), m_invm_point2(invm_point2), m_resting_len(resting_len) { }
  virtual Type getType() const { return DISTANCE; }
  virtual void enforce(NPMatrixd& x) {
    if (m_invm_point1 == 0 && m_invm_point2 == 0) return; // TODO: tolerance?
    Vector3d dir = x.row(m_i_point1) - x.row(m_i_point2);
    double norm = dir.norm();
    double weight = m_invm_point1 / (m_invm_point1 + m_invm_point2);
    dir *= (1. - m_resting_len/norm);
    x.row(m_i_point1) += -weight * dir;
    x.row(m_i_point2) += (1. - weight) * dir;
  }
};

class Cloth::Impl {
public:
  Impl(const NPMatrixd& init_x, const NPMatrixd& m, const SimulationParams& sim_params) {
    m_next_constraint_id = 0;

    // initial positions
    m_num_nodes = init_x.rows();
    if (init_x.cols() != 3) {
      throw std::runtime_error("Number of columns of init_x must be 3");
    }
    m_x = m_tmp_x = init_x;

    // initial velocities
    m_v.resize(m_x.rows(), m_x.cols());
    m_v.setZero();

    // node masses
    if (m.rows() != m_num_nodes || m.cols() != 1) {
      throw std::runtime_error("m must have shape Nx1");
    }
    m_invm.resize(m.size());
    for (int i = 0; i < m_invm.size(); ++i) {
      m_invm(i) = m(i) == 0 ? 0 : 1./m(i);
    }

    // externally applied forces
    m_f.resize(m_x.rows(), m_x.cols());
    m_f.setZero();

    m_sim_params = sim_params;
  }

  void step() {
    for (int i = 0; i < m_num_nodes; ++i) {
      m_v.row(i) += m_sim_params.dt * m_invm(i) * (m_sim_params.gravity + m_f.row(i));
    }

    // damp velocities here

    m_tmp_x = m_x + m_sim_params.dt*m_v;

    for (int iter = 0; iter < m_sim_params.solver_iters; ++iter) {
      for (int c = 0; c < m_constraints.size(); ++c) {
        if (m_constraint_on[m_constraints[c]->m_id]) {
          m_constraints[c]->enforce(m_tmp_x);
        }
      }
    }

    m_v = (m_tmp_x - m_x) / m_sim_params.dt;

    m_x = m_tmp_x;

    // velocity update here
  }

  int add_constraint(boost::shared_ptr<PositionConstraint> cnt) {
    cnt->m_id = m_next_constraint_id++;
    m_constraints.push_back(cnt);
    m_constraint_on.push_back(true);
    return cnt->m_id;
  }
  void disable_constraint(int i) { m_constraint_on[i] = false; }
  void enable_constraint(int i) { m_constraint_on[i] = true; }

private:
  friend class Cloth;

  SimulationParams m_sim_params;

  int m_num_nodes;
  NPMatrixd m_x, m_v, m_tmp_x, m_f;
  VectorXd m_invm;

  int m_next_constraint_id;
  vector<boost::shared_ptr<PositionConstraint> > m_constraints;
  vector<bool> m_constraint_on;
};


Cloth::Cloth(const NPMatrixd& init_x, const NPMatrixd& m, const SimulationParams& sim_params)
  : m_impl(new Impl(init_x, m, sim_params)) { }
Cloth::~Cloth() { delete m_impl; }

void Cloth::step() { m_impl->step(); }
py::object Cloth::get_node_positions() const { return m_impl->m_x.ndarray(); }

int Cloth::add_anchor_constraint(int i_point, const NPMatrixd& anchor_pos) {
  m_impl->m_invm(i_point) = 0;
  return m_impl->add_constraint(boost::make_shared<AnchorConstraint>(i_point, anchor_pos));
}
int Cloth::add_distance_constraint(int i_point1, int i_point2, double resting_len) {
  return m_impl->add_constraint(boost::make_shared<DistanceConstraint>(i_point1, i_point2, m_impl->m_invm(i_point1), m_impl->m_invm(i_point2), resting_len));
}

void Cloth::disable_constraint(int i) { m_impl->disable_constraint(i); }
void Cloth::enable_constraint(int i) { m_impl->enable_constraint(i); }

} // namespace tracking
