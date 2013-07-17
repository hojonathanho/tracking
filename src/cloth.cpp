#include "cloth.hpp"

#include <iostream>
using namespace std;

#include <Eigen/Core>
using namespace Eigen;

namespace tracking {

class Constraint {
};

class Cloth::Impl {
public:
  Impl(const NPMatrixd& init_x, const NPMatrixd& m, const SimulationParams& sim_params) {
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
    cout << "constructed with dt, gravity = " << sim_params.dt << ' ' << sim_params.gravity.transpose() << endl;
  }

  void step() {
    cout << "stepping, gravity = " << m_sim_params.gravity << endl;
    for (int i = 0; i < m_num_nodes; ++i) {
      m_v.row(i) += m_sim_params.dt * m_invm(i) * (m_sim_params.gravity + m_f.row(i));
    }

    // damp velocities here

    cout << "a" << endl;
    m_tmp_x = m_x + m_sim_params.dt*m_v;

    // constraints here

    cout << "b" << endl;
    m_v = (m_tmp_x - m_x) / m_sim_params.dt;
    cout << "c" << endl;
    m_x = m_tmp_x;

    // velocity update here
  }

private:
  friend class Cloth;

  SimulationParams m_sim_params;

  int m_num_nodes;
  NPMatrixd m_x, m_v, m_tmp_x, m_f;
  VectorXd m_invm;
};


Cloth::Cloth(const NPMatrixd& init_x, const NPMatrixd& m, const SimulationParams& sim_params)
  : m_impl(new Impl(init_x, m, sim_params)) {
cout << "other constructed with dt, gravity = " << sim_params.dt << ' ' << sim_params.gravity<< endl;

  }
Cloth::~Cloth() { delete m_impl; }

void Cloth::step() { m_impl->step(); }
py::object Cloth::getNodePositions() const { return m_impl->m_x.ndarray(); }

} // namespace tracking
