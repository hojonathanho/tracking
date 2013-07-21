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

static const double BENDING_CONSTRAINT_DENOM_TOL = 1e-10;

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
  return btVector3((btScalar) v(0), (btScalar) v(1), (btScalar) v(2));
}

template<typename Derived1, typename Derived2, typename Derived3>
static inline btDbvtVolume volumeOf(const MatrixBase<Derived1> &x1, const MatrixBase<Derived2> &x2, const MatrixBase<Derived3> &x3) {
  const btVector3 pts[] = { toBtVector3(x1), toBtVector3(x2), toBtVector3(x3) };
  return btDbvtVolume::FromPoints(pts, 3);
}

// Eigen-ized from btSoftBody::RayFromToCaster::rayFromToTriangle
static inline double rayFromToTriangle(
    const Vector3d& rayFrom,
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
  enum Type { ANCHOR=0, PLANE, DISTANCE, BENDING };
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

struct DistanceConstraint : public PositionConstraint {
  int m_i_point1, m_i_point2;
  double m_resting_len;
  bool m_active;
  double m_invm_ratio;
  double m_stiffness;

  DistanceConstraint(int i_point1, int i_point2, double invm_point1, double invm_point2, double resting_len, double stiffness) :
    m_i_point1(i_point1), m_i_point2(i_point2), m_resting_len(resting_len),
    m_active(invm_point1 != 0 || invm_point2 != 0),
    m_invm_ratio(invm_point1 / (invm_point1 + invm_point2)),
    m_stiffness(stiffness)
  { }

  virtual Type getType() const { return DISTANCE; }

  virtual void enforce(NPMatrixd& x) {
    if (!m_active) return;
    Vector3d dir = x.row(m_i_point1) - x.row(m_i_point2);
    dir *= (1. - m_resting_len/dir.norm());
    x.row(m_i_point1) += -m_stiffness * m_invm_ratio * dir;
    x.row(m_i_point2) += m_stiffness * (1. - m_invm_ratio) * dir;
  }
};

struct BendingConstraint : public PositionConstraint {
  vector<int> m_i_point;
  vector<double> m_invm;
  double m_resting_angle;
  double m_stiffness;

  BendingConstraint(const vector<int>& i_point, const vector<double>& invm, double resting_angle, double stiffness)
      : m_i_point(i_point), m_invm(invm), m_resting_angle(resting_angle), m_stiffness(stiffness)
  {
    assert(i_point.size() == 4);
    assert(invm.size() == 4);
  }

  virtual Type getType() const { return BENDING; }

  virtual void enforce(NPMatrixd& x) {
    Vector3d p2 = x.row(m_i_point[1]) - x.row(m_i_point[0]);
    Vector3d p3 = x.row(m_i_point[2]) - x.row(m_i_point[0]);
    Vector3d p4 = x.row(m_i_point[3]) - x.row(m_i_point[0]);

    Vector3d n1 = p2.cross(p3); double norm23 = n1.norm(); n1 /= norm23;
    Vector3d n2 = p2.cross(p4); double norm24 = n2.norm(); n2 /= norm24;
    double d = n1.dot(n2);

    Eigen::Matrix<double, 4, 3> q;
    q.row(2) = (p2.cross(n2) + n1.cross(p2)*d)/norm23;
    q.row(3) = (p2.cross(n1) + n2.cross(p2)*d)/norm24;
    q.row(1) = -(p3.cross(n2) + n1.cross(p3)*d)/norm23 - (p4.cross(n1) + n2.cross(p4)*d)/norm24;
    q.row(0) = -q.row(1) - q.row(2) - q.row(3);

    double denom = 0;
    for (int i = 0; i < 4; ++i) {
      denom += m_invm[i] * q.row(i).squaredNorm();
    }
    if (fabs(denom) < BENDING_CONSTRAINT_DENOM_TOL) {
      return;
    }
    double z = -m_stiffness * sqrt(1. - d*d) * (acos(d) - m_resting_angle) / denom;

    for (int i = 0; i < 4; ++i) {
      x.row(m_i_point[i]) += z * m_invm[i] * q.row(i);
    }
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
    if (m_sim_params.stretching_stiffness < 0 || m_sim_params.stretching_stiffness > 1) {
      throw std::runtime_error("stretching_stiffness must be in [0, 1]");
    }
    if (m_sim_params.bending_stiffness < 0 || m_sim_params.bending_stiffness > 1) {
      throw std::runtime_error("bending_stiffness must be in [0, 1]");
    }
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
    m_f += -m_sim_params.damping * m_v; // simple damping
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


  struct Collider : public btDbvt::ICollide {
    const Impl* const m_impl;
    const Vector3d m_ray_from, m_ray_to, m_ray_normalized_dir;
    double m_filter_max_dist;

    struct Result { int i_tri; double dist; };
    vector<Result> m_results;

    Collider(const Impl* impl, const Vector3d& ray_from, const Vector3d& ray_to, double filter_max_dist)
      : m_impl(impl), m_ray_from(ray_from), m_ray_to(ray_to),
        m_ray_normalized_dir((ray_to - ray_from).normalized()),
        m_filter_max_dist(filter_max_dist)
    { }

    void Process(const btDbvtNode* leaf) {
      int i_tri = reinterpret_cast<uintptr_t> (leaf->data);
      assert(0 <= i_tri && i_tri < m_impl->m_triangles.rows());
      double t = rayFromToTriangle(
        m_ray_from, m_ray_normalized_dir,
        m_impl->m_x.row(m_impl->m_triangles(i_tri,0)),
        m_impl->m_x.row(m_impl->m_triangles(i_tri,1)),
        m_impl->m_x.row(m_impl->m_triangles(i_tri,2)),
        DBL_MAX
      );
      if (t > 0 && t < m_filter_max_dist) {
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
  };

  // returns the index of the nearest triangle collided, or -1 if no collisions
  vector<Collider::Result> triangle_ray_test(const Vector3d &ray_from, const Vector3d &ray_to, double filter_max_dist) const {
    if (m_tri_dbvt.empty()) {
      throw std::runtime_error("Ray test requested, but no triangles declared");
    }

    Collider collider(this, ray_from, ray_to, filter_max_dist);

    btDbvt::rayTest(m_tri_dbvt.m_root, toBtVector3(ray_from), toBtVector3(ray_to), collider);
    collider.SortResults();
    return collider.m_results;
  }

  int triangle_ray_test_against_single_node(const Vector3d& ray_from, int i_node) const {
    double max_dist = (ray_from - m_x.row(i_node).transpose()).norm() - 1e-5;
    vector<Collider::Result> results = triangle_ray_test(ray_from, m_x.row(i_node), max_dist);
    if (results.size() == 0) {
      return -1;
    }

    int i_tri_collided = -2;
    for (int i = 0; i < results.size(); ++i) {
      // ignore collision if collided triangle contains the node as one of its vertices
      if (m_triangles.row(results[i].i_tri).cwiseEqual(i_node).any()) continue;
      i_tri_collided = results[i].i_tri; break;
    }

    return i_tri_collided;
  }

  vector<int> triangle_ray_test_against_nodes(const Vector3d &ray_from) const {
    vector<int> out(m_num_nodes);
    for (int i = 0; i < m_num_nodes; ++i) {
      out[i] = triangle_ray_test_against_single_node(ray_from, i);
    }
    return out;
  }


  double calc_stiffness_factor(double k) const {
    return 1. - pow(1. - k, 1. / m_sim_params.solver_iters);
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
// int MassSystem::triangle_ray_test(const Vector3d &ray_from, const Vector3d &ray_to) const { return m_impl->triangle_ray_test(ray_from, ray_to, DBL_MAX); } // TODO: arg for max
vector<int> MassSystem::triangle_ray_test_against_nodes(const Vector3d &ray_from) const { return m_impl->triangle_ray_test_against_nodes(ray_from); }

int MassSystem::add_anchor_constraint(int i_point, const NPMatrixd& anchor_pos) {
  m_impl->m_invm(i_point) = 0;
  return m_impl->add_constraint(boost::make_shared<AnchorConstraint>(i_point, anchor_pos));
}
int MassSystem::add_distance_constraint(int i_point1, int i_point2, double resting_len) {
  double s = m_impl->calc_stiffness_factor(m_impl->m_sim_params.stretching_stiffness);
  return m_impl->add_constraint(boost::make_shared<DistanceConstraint>(i_point1, i_point2, m_impl->m_invm(i_point1), m_impl->m_invm(i_point2), resting_len, s));
}
int MassSystem::add_plane_constraint(int i_point, const Vector3d& plane_point, const Vector3d& plane_normal) {
  return m_impl->add_constraint(boost::make_shared<PlaneConstraint>(i_point, plane_point, plane_normal));
}
int MassSystem::add_bending_constraint(int i1, int i2, int i3, int i4, double resting_angle) {
  vector<int> i_point(4);
  i_point[0] = i1; i_point[1] = i2; i_point[2] = i3; i_point[3] = i4;
  vector<double> invm(4);
  invm[0] = m_impl->m_invm(i1); invm[1] = m_impl->m_invm(i2); invm[2] = m_impl->m_invm(i3); invm[3] = m_impl->m_invm(i4);
  double stiffness = m_impl->calc_stiffness_factor(m_impl->m_sim_params.bending_stiffness);
  return m_impl->add_constraint(boost::make_shared<BendingConstraint>(i_point, invm, resting_angle, stiffness));
}

void MassSystem::disable_constraint(int i) { m_impl->disable_constraint(i); }
void MassSystem::enable_constraint(int i) { m_impl->enable_constraint(i); }
void MassSystem::randomize_constraints() { m_impl->randomize_constraints(); }


} // namespace tracking
