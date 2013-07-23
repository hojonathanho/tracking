import numpy as np
import clouds
import time

def norms(x, axis=0):
  return np.sqrt((x**2).sum(axis=axis))

def mvn_densities(x_nd, m_kd, cov_dd):
  """
  Given an array of N D-dimensional points, K means, and a single covariance matrix,
    returns a NxK array of Gaussian densities
  """

  n, k, d = x_nd.shape[0], m_kd.shape[0], m_kd.shape[1]
  assert x_nd.shape[1] == d and cov_dd.shape == (d, d)

  cov_det, cov_inv = np.linalg.det(cov_dd), np.linalg.inv(cov_dd)

  diffs = (x_nd[:,None,:] - m_kd[None,:,:]).reshape((-1, 3))

  out = np.exp(-.5 * (diffs.dot(cov_inv.T) * diffs).sum(axis=-1))
  out /= np.sqrt(np.power(2.*np.pi, d) * cov_det)
  return out.reshape((n, k))

def spherical_mvn_densities(x_nd, m_kd, cov):
  """
  Given an array of N D-dimensional points, K means, and a single covariance matrix,
    returns a NxK array of Gaussian densities
  """

  n, k, d = x_nd.shape[0], m_kd.shape[0], m_kd.shape[1]
  assert x_nd.shape[1] == d

  diffs = (x_nd[:,None,:] - m_kd[None,:,:]).reshape((-1, 3))
  out = np.exp(-.5/cov * (diffs*diffs).sum(axis=-1))
  out /= np.power(2.*np.pi*cov, d/2.)
  return out.reshape((n, k))

class Timing(object):
  t_total = 0.
  t_total_visibility = 0.
  t_total_corr = 0.
  t_total_forces = 0.
  t_total_physics  = 0.

class Tracker(object):
  pnoise = .01
  pinvisible = .1
  sigma = .0001
  force_lambda = 100
  num_em_iters = 20
  depth_occlude_tol = .03

  def __init__(self, tracked_obj):
    self.tracked_obj = tracked_obj

  def set_input(self, cloud_xyz, depth, T_w_k):
    self.cloud_xyz, self.depth, self.T_w_k = cloud_xyz, depth, T_w_k

  def calc_visibility(self):
    # Calculate visibility
    # check for nodes occluded by other nodes by raycasting from the camera
    t_begin_visibility = time.time()
    model_xyz = self.tracked_obj.get_node_positions()
    raytest_results = np.asarray(self.tracked_obj.sys.triangle_ray_test_against_nodes(self.T_w_k[:3,3]))
    occluded_by_model = raytest_results >= 0
    visibility = np.ones_like(occluded_by_model, dtype=float)
    visibility[occluded_by_model] = self.pinvisible
    # check if depth image contains a point significantly in front of a model node
    dist_cam2node = norms(self.T_w_k[:3,3][None,:] - model_xyz, axis=1)
    T_k_w = np.linalg.inv(self.T_w_k)
    depth_cam2node = clouds.lookup_depth_by_xyz(self.depth, model_xyz.dot(T_k_w[:3,:3].T) + T_k_w[:3,3])
    occluded_in_depth_img = depth_cam2node < (dist_cam2node - self.depth_occlude_tol)
    visibility[occluded_in_depth_img] = self.pinvisible
    Timing.t_total_visibility += time.time() - t_begin_visibility
    return visibility, occluded_by_model, occluded_in_depth_img

  def calc_correspondences(self, visibility):
    # Calculate expected correspondences
    t_begin_corr = time.time()
    model_xyz = self.tracked_obj.get_node_positions()
    alpha_nk = spherical_mvn_densities(self.cloud_xyz, model_xyz, self.sigma) * visibility[None,:]
    alpha_nk /= (alpha_nk.sum(axis=1) + self.pnoise)[:,None]
    Timing.t_total_corr += time.time() - t_begin_corr
    return alpha_nk

  def calc_forces(self, alpha_nk):
    # Calculate forces
    t_begin_forces = time.time()
    model_xyz = self.tracked_obj.get_node_positions()
    force_kd = self.force_lambda * (alpha_nk[:,:,None] * (self.cloud_xyz[:,None,:] - model_xyz[None,:,:])).sum(axis=0)
    Timing.t_total_forces += time.time() - t_begin_forces
    return force_kd

  def step(self, return_data=False):
    for em_iter in range(self.num_em_iters):
      print 'EM iteration %d/%d' % (em_iter+1, self.num_em_iters)
      t_begin = time.time()

      visibility_k, occluded_by_model, occluded_in_depth_img = self.calc_visibility()
      alpha_nk = self.calc_correspondences(visibility_k)
      force_kd = self.calc_forces(alpha_nk)

      t_begin_physics = time.time()
      self.tracked_obj.sys.apply_forces(force_kd)
      self.tracked_obj.step()
      Timing.t_total_physics += time.time() - t_begin_physics

      Timing.t_total += time.time() - t_begin
      if em_iter == self.num_em_iters-1:
        print 'Timing:'
        print '\tvisibility: %f (%f%%)' % (Timing.t_total_visibility, Timing.t_total_visibility/Timing.t_total*100.)
        print '\tcorr: %f (%f%%)' % (Timing.t_total_corr, Timing.t_total_corr/Timing.t_total*100.)
        print '\tforces: %f (%f%%)' % (Timing.t_total_forces, Timing.t_total_forces/Timing.t_total*100.)
        print '\tphysics: %f (%f%%)' % (Timing.t_total_physics, Timing.t_total_physics/Timing.t_total*100.)
        print '\ttotal: %f' % Timing.t_total

    if return_data:
      return {
        'occluded_in_depth_img': occluded_in_depth_img,
        'occluded_by_model': occluded_by_model,
        'force_kd': force_kd
      }


import unittest
class Tests(unittest.TestCase):
  def test_mvn_densities(self):
    x_nd = np.array([[0,0,0], [1,1,1], [1,1,2]])
    m_kd = np.array([[1,1,1], [-1,0,6]])
    cov_dd = np.diag((3, 2, 1))
    out = mvn_densities(x_nd, m_kd, cov_dd)
    for i in range(len(x_nd)):
      for j in range(len(m_kd)):
        self.assertEqual(out[i,j], mvn_densities(x_nd[[i]], m_kd[[j]], cov_dd))

  def test_mvn_densities_spherical(self):
    x_nd = np.array([[0,0,0], [1,1,1], [1,1,2]])
    m_kd = np.array([[1,1,1], [-1,0,6]])
    cov = 3.1415926
    cov_dd = np.diag((cov, cov, cov))
    self.assertTrue(np.allclose(mvn_densities(x_nd, m_kd, cov_dd), spherical_mvn_densities(x_nd, m_kd, cov)))


if __name__ == '__main__':
  unittest.main()
