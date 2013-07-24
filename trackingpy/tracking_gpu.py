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
    return (32, 32, 1)

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

__global__ void calc_corr_normalization(float *a_nk, float pnoise, int N, int K, float *out_n) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= N) return;

  out_n[i] = pnoise;
  for (int j = 0; j < K; ++j) {
    out_n[i] += a_nk[i*K + j];
  }
  out_n[i] = 1.f/out_n[i];
}

__global__ void calc_forces(float lambda, float *alpha_nk, float *x_nd, float *m_kd, int N, int K, float *out_nkd) {
  int i = threadIdx.y + blockIdx.y*blockDim.y;
  int j = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= N || j >= K) return;

  for (int d = 0; d < D; ++d) {
    out_nkd[i*K*D + j*D + d] = lambda * alpha_nk[i*K + j] * (x_nd[i*D + d] - m_kd[j*D + d]);
  }
}

__global__ void sum_forces_across_cloud(float *in_nkd, int N, int K, float *out_kd) {
  int k = threadIdx.y + blockIdx.y*blockDim.y;
  int d = threadIdx.x + blockIdx.x*blockDim.x;
  if (k >= K || d >= D) return;

  out_kd[k*D + d] = 0.f;
  for (int i = 0; i < N; ++i) {
    out_kd[k*D + d] += in_nkd[i*K*D + k*D + d];
  }
}

  """)

  spherical_mvn = mod.get_function('spherical_mvn')
  spherical_mvn.prepare('PPfiiP')

  calc_corr_normalization = mod.get_function('calc_corr_normalization')
  calc_corr_normalization.prepare('PfiiP')

  mult_rowwise = mod.get_function('mult_rowwise')
  mult_rowwise.prepare('PPii')

  mult_colwise = mod.get_function('mult_colwise')
  mult_colwise.prepare('PPii')

  calc_forces = mod.get_function('calc_forces')
  calc_forces.prepare('fPPPiiP')

  sum_forces_across_cloud = mod.get_function('sum_forces_across_cloud')
  sum_forces_across_cloud.prepare('PiiP')


class GPUTracker(tracking.Tracker):
  def __init__(self, tracked_obj):
    tracking.Tracker.__init__(self, tracked_obj)
    self.K, self.D = self.tracked_obj.get_num_nodes(), 3
    self.model_xyz_gpu = gpuarray.empty((self.K, self.D), np.float32)
    self.visibilities_gpu = gpuarray.empty(self.K, np.float32)
    self.force_kd_gpu = gpuarray.empty((self.K, self.D), np.float32)

    # max number of cloud points
    self.max_N = 10000
    self.cloud_xyz_gpu = gpuarray.empty((self.max_N, self.D), np.float32)
    self.force_cache_nkd_gpu = gpuarray.empty((self.max_N, self.K, self.D), np.float32)
    self.alpha_nk_gpu = gpuarray.empty((self.max_N, self.K), np.float32)
    self.normalization_n_gpu = gpuarray.empty(self.max_N, np.float32)

  def set_input(self, cloud_xyz, depth, T_w_k):
    tracking.Tracker.set_input(self, cloud_xyz, depth, T_w_k)
    self.N, self.K, self.D = len(cloud_xyz), self.tracked_obj.get_num_nodes(), cloud_xyz.shape[1]
    if self.N > self.max_N:
      print 'WARNING: more points in cloud (%d) than points in pre-allocated memory (%d). Truncating cloud.' % (self.N, self.max_N)
      cloud_xyz = cloud_xyz[:self.max_N]
      self.N = self.max_N
    cuda.memcpy_htod(self.cloud_xyz_gpu.gpudata, cloud_xyz.astype(np.float32))

  def calc_correspondences(self, visibility):
    # Calculate expected correspondences
    cuda.memcpy_htod(self.model_xyz_gpu.gpudata, self.tracked_obj.get_node_positions().astype(np.float32))
    cuda.memcpy_htod(self.visibilities_gpu.gpudata, visibility.astype(np.float32))
    GPUMethods.spherical_mvn.prepared_call(
      GPUArgs.grid(self.N, self.K), GPUArgs.block(),
      self.cloud_xyz_gpu.gpudata, self.model_xyz_gpu.gpudata, self.sigma, self.N, self.K, self.alpha_nk_gpu.gpudata
    )
    GPUMethods.mult_rowwise.prepared_call(
      GPUArgs.grid(self.N, self.K), GPUArgs.block(),
      self.alpha_nk_gpu.gpudata, self.visibilities_gpu.gpudata, self.N, self.K
    )
    GPUMethods.calc_corr_normalization.prepared_call(
      (int(np.ceil(self.N/64.)), 1), (64, 1, 1),
      self.alpha_nk_gpu.gpudata, self.pnoise, self.N, self.K, self.normalization_n_gpu.gpudata
    )
    GPUMethods.mult_colwise.prepared_call(
      GPUArgs.grid(self.N, self.K), GPUArgs.block(),
      self.alpha_nk_gpu.gpudata, self.normalization_n_gpu.gpudata, self.N, self.K
    )
    return self.alpha_nk_gpu

  def calc_forces(self, alpha_nk_gpu):
    # Calculate forces
    GPUMethods.calc_forces.prepared_call(
      GPUArgs.grid(self.N, self.K), GPUArgs.block(),
      self.force_lambda, alpha_nk_gpu.gpudata, self.cloud_xyz_gpu.gpudata, self.model_xyz_gpu.gpudata, self.N, self.K, self.force_cache_nkd_gpu.gpudata,
    )
    GPUMethods.sum_forces_across_cloud.prepared_call(
      (1, int(np.ceil(self.K/64.))), (self.D, 64, 1),
      self.force_cache_nkd_gpu.gpudata, self.N, self.K, self.force_kd_gpu.gpudata,
    )
    force_kd = self.force_kd_gpu.get().astype(float)
    return force_kd
