#include "mass_system.hpp"

#include <iostream>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>
using namespace Eigen;

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <cfloat>
#include "BulletCollision/BroadphaseCollision/btDbvt.h"

namespace tracking {


////////// Utility functions //////////

template<typename Derived>
static inline Matrix3d crossprod(const MatrixBase<Derived> &v) {
  Matrix3d m;
  m <<     0, -v(2),  v(1),
        v(2),     0, -v(0),
       -v(1),  v(0),     0;
  return m;
}

template<typename Derived>
static inline btVector3 toBtVector3(const MatrixBase<Derived> &v) {
  return btVector3(v(0), v(1), v(2));
}

template<typename Derived1, typename Derived2, typename Derived3>
static inline btDbvtVolume volumeOf(const MatrixBase<Derived1> &x1, const MatrixBase<Derived2> &x2, const MatrixBase<Derived3> &x3) {
  const btVector3 pts[] = { toBtVector3(x1), toBtVector3(x2), toBtVector3(x3) };
  return btDbvtVolume::FromPoints(pts, 3);
}

// Eigen-ized from btSoftBody::RayFromToCaster::rayFromToTriangle
static inline double rayFromToTriangle(
    const Vector3d& rayFrom,
    const Vector3d& rayTo,
    const Vector3d& rayNormalizedDirection,
    const Vector3d& a,
    const Vector3d& b,
    const Vector3d& c,
    double maxt) {

  static const double ceps=-DBL_EPSILON*10;
  static const double teps=DBL_EPSILON*10;

  const Vector3d n = (b-a).cross(c-a);
  const double d = a.dot(n);
  const double den = rayNormalizedDirection.dot(n);
  if (fabs(den) >= DBL_EPSILON) {
    const double num = rayFrom.dot(n) - d;
    const double t = -num/den;
    if(t > teps && t < maxt) {
      const Vector3d hit = rayFrom + rayNormalizedDirection*t;
      if(n.dot((a-hit).cross(b-hit)) > ceps &&
         n.dot((b-hit).cross(c-hit)) > ceps &&
         n.dot((c-hit).cross(a-hit)) > ceps) {
        return t;
      }
    }
  }
  return -1;
}

static vector<int> randomPermutation(const vector<int>& in) {
  // use the numpy random number generator
  vector<int> out(in.size());
  py::list l = py::extract<py::list>(GetNumPyMod().attr("random").attr("permutation")(in).attr("tolist")());
  for (int i = 0; i < in.size(); ++i) {
    out[i] = py::extract<int>(l[i]);
  }
  return out;
}

////////// Constraint structures //////////

struct PositionConstraint {
  enum Type { ANCHOR=0, DISTANCE, PLANE };
  int m_id;
  bool m_enabled;
  virtual Type getType() const = 0;
  virtual void enforce(NPMatrixd& x) = 0; // x is the positions of the nodes
};
typedef boost::shared_ptr<PositionConstraint> PositionConstraintPtr;

struct AnchorConstraint : public PositionConstraint {
  int m_i_point; Vector3d m_anchor_pos;

  AnchorConstraint(int i_point, const Vector3d &anchor_pos) : m_i_point(i_point), m_anchor_pos(anchor_pos) { }

  virtual Type getType() const { return ANCHOR; }

  virtual void enforce(NPMatrixd& x) {
    x.row(m_i_point) = m_anchor_pos;
  }
};

struct DistanceConstraint : public PositionConstraint {
  int m_i_point1, m_i_point2;
  double m_resting_len;
  bool m_active;
  double m_invm_ratio;

  DistanceConstraint(int i_point1, int i_point2, double invm_point1, double invm_point2, double resting_len) :
    m_i_point1(i_point1), m_i_point2(i_point2), m_resting_len(resting_len),
    m_active(invm_point1 != 0 || invm_point2 != 0),
    m_invm_ratio(invm_point1 / (invm_point1 + invm_point2))
  { }

  virtual Type getType() const { return DISTANCE; }

  virtual void enforce(NPMatrixd& x) {
    if (!m_active) return;
    Vector3d dir = x.row(m_i_point1) - x.row(m_i_point2);
    dir *= (1. - m_resting_len/dir.norm());
    x.row(m_i_point1) += -m_invm_ratio * dir;
    x.row(m_i_point2) += (1. - m_invm_ratio) * dir;
  }
};

struct PlaneConstraint : public PositionConstraint {
  int m_i_point;
  Vector3d m_plane_point;
  Vector3d m_plane_normal;

  PlaneConstraint(int i_point, const Vector3d& plane_point, const Vector3d& plane_normal) :
    m_i_point(i_point), m_plane_point(plane_point), m_plane_normal(plane_normal.normalized()) { }

  virtual Type getType() const { return PLANE; }

  virtual void enforce(NPMatrixd& x) {
    Vector3d d = x.row(m_i_point).transpose() - m_plane_point;
    double dot = m_plane_normal.dot(d);
    if (dot >= 0) return;
    x.row(m_i_point) = m_plane_point + d - dot*m_plane_normal;
  }
};




////////// Position-based dynamics implementation //////////

struct MassSystem::Impl {
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
    m_mass = m;
    m_total_mass = m_mass.sum();
    m_invm.resize(m.size());
    for (int i = 0; i < m_invm.size(); ++i) {
      m_invm(i) = m(i) == 0 ? 0 : 1./m(i);
    }

    // externally applied forces
    m_f.resize(m_x.rows(), m_x.cols());
    m_f.setZero();

    m_sim_params = sim_params;
    // if (m_sim_params.damping < 0 || m_sim_params.damping > 1) {
    //   throw std::runtime_error("damping must be in [0, 1]");
    // }
  }


  ///// Position-based dynamics /////
  void apply_damping() {
    Vector3d x_cm(0, 0, 0);
    for (int i = 0; i < m_num_nodes; ++i) {
      x_cm += m_x.row(i) * m_mass(i);
    }
    x_cm /= m_total_mass;

    Vector3d v_cm(0, 0, 0);
    for (int i = 0; i < m_num_nodes; ++i) {
      v_cm += m_v.row(i) * m_mass(i);
    }
    v_cm /= m_total_mass;

    Vector3d L(0, 0, 0);
    for (int i = 0; i < m_num_nodes; ++i) {
      Vector3d r = m_x.row(i).transpose() - x_cm;
      L += m_mass(i) * r.cross((Vector3d) m_v.row(i));
    }

    Matrix3d I(Matrix3d::Zero());
    for (int i = 0; i < m_num_nodes; ++i) {
      Matrix3d cp(crossprod(m_x.row(i).transpose() - x_cm));
      I += cp * cp.transpose() * m_mass(i);
    }

    Vector3d omega = I.inverse() * L;

    for (int i = 0; i < m_num_nodes; ++i) {
      Vector3d r = m_x.row(i).transpose() - x_cm;
      m_v.row(i) += m_sim_params.damping * (v_cm + omega.cross(r) - m_v.row(i).transpose());
    }
  }

  void step() {
    // velocity step
    m_f += -m_sim_params.damping * m_v;
    for (int i = 0; i < m_num_nodes; ++i) {
      m_v.row(i) += m_sim_params.dt * m_invm(i) * (m_sim_params.gravity.transpose() + m_f.row(i));
    }
    // apply_damping();

    // position step, ignoring constraints
    m_tmp_x = m_x + m_sim_params.dt*m_v;

    // satisfy constraints
    for (int iter = 0; iter < m_sim_params.solver_iters; ++iter) {
      for (int c = 0; c < m_constraint_ordering.size(); ++c) {
        PositionConstraint& cnt = *m_constraints[m_constraint_ordering[c]];
        if (cnt.m_enabled) {
          cnt.enforce(m_tmp_x);
        }
      }
    }

    // retroactively set velocities
    m_v = (m_tmp_x - m_x) / m_sim_params.dt;
    m_x = m_tmp_x;

    // velocity update here

    // update acceleration structures
    update_accel();

    // cleanup
    m_f.setZero();
  }

  void apply_forces(const NPMatrixd& f) {
    m_f += f;
  }

  ///// Constraint methods /////

  int add_constraint(PositionConstraintPtr cnt) {
    cnt->m_id = m_next_constraint_id++;
    cnt->m_enabled = true;
    m_constraints.push_back(cnt);
    m_constraint_ordering.push_back(cnt->m_id);
    return cnt->m_id;
  }
  void disable_constraint(int i) { m_constraints[i]->m_enabled = false; }
  void enable_constraint(int i) { m_constraints[i]->m_enabled = true; }
  void randomize_constraints() { m_constraint_ordering = randomPermutation(m_constraint_ordering); }


  ///// Triangles/faces methods /////

  // TODO: scale by margin?
  btDbvtVolume calc_triangle_volume(int i_tri) const {
    return volumeOf(m_x.row(m_triangles(i_tri,0)), m_x.row(m_triangles(i_tri,1)), m_x.row(m_triangles(i_tri,2)));
  }
  Vector3d calc_triangle_velocity(int i_tri) const {
    return (m_v.row(m_triangles(i_tri,0)) + m_v.row(m_triangles(i_tri,1)) + m_v.row(m_triangles(i_tri,2))) / 3.;
  }

  void declare_triangles(const NPMatrixi& triangles) {
    if (triangles.rows() == 0 || triangles.cols() != 3) {
      throw std::runtime_error("Input triangles is empty or doesn't have 3 columns");
    }
    m_triangles = triangles;

    // populate the triangle tree
    m_tri_dbvt.clear();
    int num_triangles = m_triangles.rows();
    m_tri_dbvt_leaves.resize(num_triangles);
    for (int i = 0; i < num_triangles; ++i) {
      btDbvtVolume vol = calc_triangle_volume(i);
      m_tri_dbvt_leaves[i] = m_tri_dbvt.insert(vol, (void*)i); // stored leaf data is the triangle index
    }
  }

  void update_accel() {
    if (!m_tri_dbvt.empty()) {
      assert(m_tri_dbvt_leaves.size() == m_triangles.rows());
      for (int i = 0; i < m_triangles.rows(); ++i) {
        // TODO: scaling
        btDbvtVolume vol = calc_triangle_volume(i);
        m_tri_dbvt.update(m_tri_dbvt_leaves[i], vol, toBtVector3(calc_triangle_velocity(i)));
      }
      m_tri_dbvt.optimizeIncremental(1);
    }
  }

  // returns the index of the nearest triangle collided, or -1 if no collisions
  int triangle_ray_test(const Vector3d &ray_from, const Vector3d &ray_to) const {
    if (m_tri_dbvt.empty()) {
      throw std::runtime_error("Ray test requested, but no triangles declared");
    }

    struct Collider : public btDbvt::ICollide {
      const Impl* const m_impl;
      const Vector3d m_ray_from, m_ray_to, m_ray_normalized_dir;

      struct Result { int i_tri; double dist; };
      vector<Result> m_results;

      Collider(const Impl* impl, const Vector3d& ray_from, const Vector3d& ray_to)
        : m_impl(impl), m_ray_from(ray_from), m_ray_to(ray_to),
          m_ray_normalized_dir((ray_to - ray_from).normalized())
      { }

      void Process(const btDbvtNode* leaf) {
        int i_tri = reinterpret_cast<uintptr_t> (leaf->data);
        assert(0 <= i_tri && i_tri < m_impl->m_triangles.rows());
        double t = rayFromToTriangle(
          m_ray_from, m_ray_to, m_ray_normalized_dir,
          m_impl->m_x.row(m_impl->m_triangles(i_tri,0)),
          m_impl->m_x.row(m_impl->m_triangles(i_tri,1)),
          m_impl->m_x.row(m_impl->m_triangles(i_tri,2)),
          DBL_MAX
        );
        if (t > 0) {
          Result res = { i_tri, t };
          m_results.push_back(res);
        }
      }

      struct ResultCmp {
        bool operator()(const Result &r1, const Result &r2) const { return r1.dist < r2.dist; }
      };
      void SortResults() {
        std::sort(m_results.begin(), m_results.end(), ResultCmp());
      }

    } collider(this, ray_from, ray_to);

    btDbvt::rayTest(m_tri_dbvt.m_root, toBtVector3(ray_from), toBtVector3(ray_to), collider);
    collider.SortResults();
    return collider.m_results.empty() ? -1 : collider.m_results[0].i_tri;
  }

  int triangle_ray_test_against_single_node(int i_node, const Vector3d& ray_from) const {
    // TODO: check that the collided triangle is in front of the node
    int i_tri_collided = triangle_ray_test(ray_from, m_x.row(i_node));
    // ignore collision if collided triangle contains the node as one of its vertices
    if (i_tri_collided == -1 || m_triangles.row(i_tri_collided).cwiseEqual(i_node).any()) {
      return -1;
    }
    return i_tri_collided;
  }

  vector<int> triangle_ray_test_against_nodes(const Vector3d &ray_from) const {
    vector<int> out(m_num_nodes);
    for (int i = 0; i < m_num_nodes; ++i) {
      out[i] = triangle_ray_test_against_single_node(i, ray_from);
    }
    return out;
  }


  SimulationParams m_sim_params;

  // State of masses
  int m_num_nodes;
  NPMatrixd m_x, m_v, m_tmp_x, m_f;
  VectorXd m_mass, m_invm; double m_total_mass;

  // Constraint data
  int m_next_constraint_id;
  vector<PositionConstraintPtr> m_constraints;
  vector<int> m_constraint_ordering;

  // Triangle (face) information
  NPMatrixi m_triangles; // shape == (num_triangles, 3), each row stores the indices of the vertices of one triangle
  btDbvt m_tri_dbvt; // acceleration structure for triangles
  vector<btDbvtNode*> m_tri_dbvt_leaves; // leaf in the triangle tree for each triangle
};



////////// Publicly-exposed methods //////////

MassSystem::MassSystem(const NPMatrixd& init_x, const NPMatrixd& m, const SimulationParams& sim_params)
  : m_impl(new Impl(init_x, m, sim_params)) { }
MassSystem::~MassSystem() { delete m_impl; }

void MassSystem::apply_forces(const NPMatrixd& f) { m_impl->apply_forces(f); }
void MassSystem::step() { m_impl->step(); }
py::object MassSystem::get_node_positions() const { return m_impl->m_x.ndarray(); }

void MassSystem::declare_triangles(const NPMatrixi& triangles) { m_impl->declare_triangles(triangles); }
int MassSystem::triangle_ray_test(const Vector3d &ray_from, const Vector3d &ray_to) const { return m_impl->triangle_ray_test(ray_from, ray_to); }
vector<int> MassSystem::triangle_ray_test_against_nodes(const Vector3d &ray_from) const { return m_impl->triangle_ray_test_against_nodes(ray_from); }

int MassSystem::add_anchor_constraint(int i_point, const NPMatrixd& anchor_pos) {
  m_impl->m_invm(i_point) = 0;
  return m_impl->add_constraint(boost::make_shared<AnchorConstraint>(i_point, anchor_pos));
}
int MassSystem::add_distance_constraint(int i_point1, int i_point2, double resting_len) {
  return m_impl->add_constraint(boost::make_shared<DistanceConstraint>(i_point1, i_point2, m_impl->m_invm(i_point1), m_impl->m_invm(i_point2), resting_len));
}
int MassSystem::add_plane_constraint(int i_point, const Vector3d& plane_point, const Vector3d& plane_normal) {
  return m_impl->add_constraint(boost::make_shared<PlaneConstraint>(i_point, plane_point, plane_normal));
}

void MassSystem::disable_constraint(int i) { m_impl->disable_constraint(i); }
void MassSystem::enable_constraint(int i) { m_impl->enable_constraint(i); }
void MassSystem::randomize_constraints() { m_impl->randomize_constraints(); }


} // namespace tracking
