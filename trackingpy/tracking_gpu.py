from trackingpy import tracking
import numpy as np
import time

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.compiler

class GPUArgs(object):
  @staticmethod
  def block():
    return (4, 4, 1)

  @staticmethod
  def grid(N, K):
    return (int(np.ceil(float(K)/GPUArgs.block()[0])), int(np.ceil(float(N)/GPUArgs.block()[1])))

class GPUMethods(object):
  mod = pycuda.compiler.SourceModule("""
#define D 3

__global__ void spherical_mvn(float *x_nd, float *m_kd, float cov, int N, int K, float *out) {
  int i = threadIdx.y + blockIdx.y*blockDim.y;
  int j = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= N || j >= K) return;

  float quadform = 0.f;
  for (int k = 0; k < D; ++k) {
    float z = x_nd[i*D + k] - m_kd[j*D + k];
    quadform += z*z;
  }

  out[i*K + j] = powf(2.f*3.141592654f*cov, -D/2.f) * expf(-.5f/cov * quadform);
}

__global__ void mult_rowwise(float *a_nk, float *b_k, int N, int K) {
  int i = threadIdx.y + blockIdx.y*blockDim.y;
  int j = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= N || j >= K) return;

  a_nk[i*K + j] *= b_k[j];
}

__global__ void mult_colwise(float *a_nk, float *b_n, int N, int K) {
  int i = threadIdx.y + blockIdx.y*blockDim.y;
  int j = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= N || j >= K) return;

  a_nk[i*K + j] *= b_n[i];
}

__global__ void calc_forces(float lambda, float *alpha_nk, float *x_nd, float *m_kd, int N, int K, float *out_nkd) {
  int i = threadIdx.y + blockIdx.y*blockDim.y;
  int j = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= N || j >= K) return;

  for (int d = 0; d < D; ++d) {
    out_nkd[i*K*D + j*D + d] = lambda * alpha_nk[i*K + j] * (x_nd[i*D + d] - m_kd[j*D + d]);
  }
}

__global__ void test_fill(float *out, int N, int K) {
  int i = threadIdx.y + blockIdx.y*blockDim.y;
  int j = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= N || j >= K) return;
  int out_idx = j + i*K;
  out[out_idx] = i*100 + j;
}
  """)

  spherical_mvn = mod.get_function('spherical_mvn')
  mult_rowwise = mod.get_function('mult_rowwise')
  mult_colwise = mod.get_function('mult_colwise')
  calc_forces = mod.get_function('calc_forces')

  @staticmethod
  def spherical_mvn_densities(x_nd, m_kd, cov, out=None):
    N, K, D = x_nd.shape[0], m_kd.shape[0], m_kd.shape[1]
    assert x_nd.shape[1] == D and isinstance(x_nd, gpuarray.GPUArray) and isinstance(m_kd, gpuarray.GPUArray) and D == 3
    if out is None:
      out = gpuarray.empty((N, K), np.float32)
    else:
      assert isinstance(out, gpuarray.GPUArray) and out.shape == (N, K)
    GPUMethods.spherical_mvn(x_nd.gpudata, m_kd.gpudata, np.float32(cov), np.int32(N), np.int32(K), out.gpudata, block=GPUArgs.block(), grid=GPUArgs.grid(N, K))
    return out



# def em_loop_gpu(tracked_obj, cloud_xyz, depth, T_w_k, return_data=False):
#   pnoise = .01
#   pinvisible = .1
#   sigma = .0001
#   force_lambda = 100
#   num_em_iters = 20
#   depth_occlude_tol = .03

#   for em_iter in range(num_em_iters):
#     print 'EM iteration %d/%d' % (em_iter+1, num_em_iters)
#     t_begin = time.time()

#     model_xyz = tracked_obj.get_node_positions()
#     # Calculate visibility
#     # check for nodes occluded by other nodes by raycasting from the camera
#     t_begin_visibility = time.time()
#     raytest_results = np.asarray(tracked_obj.sys.triangle_ray_test_against_nodes(T_w_k[:3,3]))
#     occluded_by_model = raytest_results >= 0
#     visibility = np.ones_like(occluded_by_model, dtype=float)
#     visibility[occluded_by_model] = pinvisible
#     # check if depth image contains a point significantly in front of a model node
#     dist_cam2node = norms(T_w_k[:3,3][None,:] - model_xyz, axis=1)
#     T_k_w = np.linalg.inv(T_w_k)
#     depth_cam2node = clouds.lookup_depth_by_xyz(depth, model_xyz.dot(T_k_w[:3,:3].T) + T_k_w[:3,3])
#     occluded_in_depth_img = depth_cam2node < (dist_cam2node - depth_occlude_tol)
#     visibility[occluded_in_depth_img] = pinvisible
#     Timing.t_total_visibility += time.time() - t_begin_visibility

#     model_xyz_gpu = gpuarray.to_gpu(model_xyz.astype(np.float32))
#     cloud_xyz_gpu = gpuarray.to_gpu(cloud_xyz.astype(np.float32))
#     visibilities_gpu = gpuarray.to_gpu(visibility.astype(np.float32))

#     # Calculate expected correspondences
#     t_begin_corr = time.time()
#     alpha_NK_gpu = spherical_mvn_densities_gpu(cloud_xyz_gpu, model_xyz_gpu, sigma)
#     mod.get_function("mult_rowwise")(alpha_NK_gpu.gpudata, visibilities_gpu.gpudata, np.int32(N), np.int32(K), block=GPUArgs.block(), grid=GPUArgs.grid(N, K))
#     norms_n = 1. / (alpha_NK_gpu.get().sum(axis=1) + pnoise)
#     mod.get_function("mult_colwise")(alpha_NK_gpu.gpudata, gpuarray.to_gpu(norms_n.astype(np.float32)).gpudata, np.int32(N), np.int32(K), block=GPUArgs.block(), grid=GPUArgs.grid(N, K))
#     Timing.t_total_corr += time.time() - t_begin_corr

#     # Calculate and apply forces
#     t_begin_forces = time.time()
#     force_kd_gpu = gpuarray.empty((N, K, D), np.float32)
#     mod.get_function("calc_forces")(np.float32(force_lambda), alpha_NK_gpu.gpudata, cloud_xyz_gpu.gpudata, model_xyz_gpu.gpudata, np.int32(N), np.int32(K), force_kd_gpu.gpudata, block=GPUArgs.block(), grid=GPUArgs.grid(N, K))
#     force_kd = force_kd_gpu.get().sum(axis=0).astype(float)
#     Timing.t_total_forces += time.time() - t_begin_forces

#     t_begin_physics = time.time()
#     tracked_obj.sys.apply_forces(force_kd)
#     tracked_obj.step()
#     Timing.t_total_physics += time.time() - t_begin_physics

#     Timing.t_total += time.time() - t_begin

#     if em_iter == num_em_iters-1:
#       print 'Timing:'
#       print '\tvisibility: %f (%f%%)' % (Timing.t_total_visibility, Timing.t_total_visibility/Timing.t_total*100.)
#       print '\tcorr: %f (%f%%)' % (Timing.t_total_corr, Timing.t_total_corr/Timing.t_total*100.)
#       print '\tforces: %f (%f%%)' % (Timing.t_total_forces, Timing.t_total_forces/Timing.t_total*100.)
#       print '\tphysics: %f (%f%%)' % (Timing.t_total_physics, Timing.t_total_physics/Timing.t_total*100.)
#       print '\ttotal: %f' % Timing.t_total

#   if return_data:
#     return {
#       'occluded_in_depth_img': occluded_in_depth_img,
#       'occluded_by_model': occluded_by_model,
#       'force_kd': force_kd
#     }

#   return None



class GPUTracker(tracking.Tracker):
  def set_input(self, cloud_xyz, depth, T_w_k):
    self.cloud_xyz, self.depth, self.T_w_k = cloud_xyz, depth, T_w_k
    self.cloud_xyz_gpu = gpuarray.to_gpu(cloud_xyz.astype(np.float32))
    self.N, self.K, self.D = len(cloud_xyz), self.tracked_obj.get_num_nodes(), cloud_xyz.shape[1]

  def calc_correspondences(self, visibility):
    # Calculate expected correspondences
    t_begin_corr = time.time()
    self.model_xyz_gpu = gpuarray.to_gpu(self.tracked_obj.get_node_positions().astype(np.float32)) # need to re-copy after every physics step
    visibilities_gpu = gpuarray.to_gpu(visibility.astype(np.float32))
    alpha_nk_gpu = GPUMethods.spherical_mvn_densities(self.cloud_xyz_gpu, self.model_xyz_gpu, self.sigma)
    GPUMethods.mult_rowwise(alpha_nk_gpu.gpudata, visibilities_gpu.gpudata, np.int32(self.N), np.int32(self.K), block=GPUArgs.block(), grid=GPUArgs.grid(self.N, self.K))
    norms_n = 1. / (alpha_nk_gpu.get().sum(axis=1) + self.pnoise)
    GPUMethods.mult_colwise(alpha_nk_gpu.gpudata, gpuarray.to_gpu(norms_n.astype(np.float32)).gpudata, np.int32(self.N), np.int32(self.K), block=GPUArgs.block(), grid=GPUArgs.grid(self.N, self.K))
    tracking.Timing.t_total_corr += time.time() - t_begin_corr
    return alpha_nk_gpu

  def calc_forces(self, alpha_nk_gpu):
    # Calculate forces
    t_begin_forces = time.time()
    force_kd_gpu = gpuarray.empty((self.N, self.K, 3), np.float32)
    GPUMethods.calc_forces(np.float32(self.force_lambda), alpha_nk_gpu.gpudata, self.cloud_xyz_gpu.gpudata, self.model_xyz_gpu.gpudata, np.int32(self.N), np.int32(self.K), force_kd_gpu.gpudata, block=GPUArgs.block(), grid=GPUArgs.grid(self.N, self.K))
    force_kd = force_kd_gpu.get().sum(axis=0).astype(float)
    tracking.Timing.t_total_forces += time.time() - t_begin_forces
    return force_kd
