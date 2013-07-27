from trackingpy import tracking
import numpy as np
import time

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.compiler

def div_up(x, y):
  return int(np.ceil(float(x)/y))

class GPUParams(object):
  max_threads_per_block = pycuda.autoinit.device.get_attribute(pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK)
  max_square_side = int(np.sqrt(max_threads_per_block))

class GPUFunctions(object):
  mod = pycuda.compiler.SourceModule("""
#define D 3

__global__ void isotropic_mvn(float *x_nd, float *m_kd, float cov, int N, int K, float *out) {
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

  class FuncWrapper(object):
    def __init__(self, mod, name, arg_spec):
      self.fn = mod.get_function(name)
      self.fn.prepare(arg_spec)
      self.grid, self.block = None, None

    def __call__(self, *args):
      assert self.grid is not None and self.block is not None
      return self.fn.prepared_call(self.grid, self.block, *args)

  isotropic_mvn = FuncWrapper(mod, 'isotropic_mvn', 'PPfiiP')
  calc_corr_normalization = FuncWrapper(mod, 'calc_corr_normalization', 'PfiiP')
  mult_rowwise = FuncWrapper(mod, 'mult_rowwise', 'PPii')
  mult_colwise = FuncWrapper(mod, 'mult_colwise', 'PPii')
  calc_forces = FuncWrapper(mod, 'calc_forces', 'fPPPiiP')
  sum_forces_across_cloud = FuncWrapper(mod, 'sum_forces_across_cloud', 'PiiP')

  @classmethod
  def set_grid_and_block_sizes(cls, N, K, D):
    square_side = min(32, GPUParams.max_square_side)

    cls.isotropic_mvn.block = (square_side, square_side, 1)
    cls.isotropic_mvn.grid = (div_up(K, cls.isotropic_mvn.block[0]), div_up(N, cls.isotropic_mvn.block[1]))

    cls.calc_corr_normalization.block = (64, 1, 1)
    cls.calc_corr_normalization.grid = (div_up(N, cls.calc_corr_normalization.block[0]), 1)

    cls.mult_rowwise.block = (square_side, square_side, 1)
    cls.mult_rowwise.grid = (div_up(K, cls.mult_rowwise.block[0]), div_up(N, cls.mult_rowwise.block[1]))

    cls.mult_colwise.block = (square_side, square_side, 1)
    cls.mult_colwise.grid = (div_up(K, cls.mult_colwise.block[0]), div_up(N, cls.mult_colwise.block[1]))

    cls.calc_forces.block = (square_side, square_side, 1)
    cls.calc_forces.grid = (div_up(K, cls.calc_forces.block[0]), div_up(N, cls.calc_forces.block[1]))

    cls.sum_forces_across_cloud.block = (D, min(64, GPUParams.max_threads_per_block//D), 1)
    cls.sum_forces_across_cloud.grid = (1, div_up(K, 64))

class GPUTracker(tracking.Tracker):
  def __init__(self, tracked_obj):
    tracking.Tracker.__init__(self, tracked_obj)
    self.K, self.D = self.tracked_obj.get_num_nodes(), 3
    self.max_N = 10000 # max number of cloud points

    # pre-allocate GPU memory
    self.model_xyz_gpu = gpuarray.empty((self.K, self.D), np.float32)
    self.visibilities_gpu = gpuarray.empty(self.K, np.float32)
    self.force_kd_gpu = gpuarray.empty((self.K, self.D), np.float32)
    # for arrays with input-dependent sizes, allocate a maximum amount
    self.cloud_xyz_gpu = gpuarray.empty((self.max_N, self.D), np.float32)
    self.force_cache_nkd_gpu = gpuarray.empty((self.max_N, self.K, self.D), np.float32)
    self.alpha_nk_gpu = gpuarray.empty((self.max_N, self.K), np.float32)
    self.normalization_n_gpu = gpuarray.empty(self.max_N, np.float32)

  def set_input(self, cloud_xyz, depth, T_w_k):
    tracking.Tracker.set_input(self, cloud_xyz, depth, T_w_k)

    # copy the input point cloud to the gpu
    self.N = len(cloud_xyz)
    if self.N > self.max_N:
      print 'WARNING: more points in cloud (%d) than points in pre-allocated memory (%d). Truncating cloud.' % (self.N, self.max_N)
      cloud_xyz = cloud_xyz[:self.max_N]
      self.N = self.max_N
    cuda.memcpy_htod(self.cloud_xyz_gpu.gpudata, cloud_xyz.astype(np.float32))

    # set block sizes of kernels
    GPUFunctions.set_grid_and_block_sizes(self.N, self.K, self.D)

  def calc_correspondences(self, visibility):
    cuda.memcpy_htod(self.model_xyz_gpu.gpudata, self.tracked_obj.get_node_positions().astype(np.float32))
    cuda.memcpy_htod(self.visibilities_gpu.gpudata, visibility.astype(np.float32))

    GPUFunctions.isotropic_mvn(self.cloud_xyz_gpu.gpudata, self.model_xyz_gpu.gpudata, self.sigma, self.N, self.K, self.alpha_nk_gpu.gpudata)
    GPUFunctions.mult_rowwise(self.alpha_nk_gpu.gpudata, self.visibilities_gpu.gpudata, self.N, self.K)
    GPUFunctions.calc_corr_normalization(self.alpha_nk_gpu.gpudata, self.pnoise, self.N, self.K, self.normalization_n_gpu.gpudata)
    GPUFunctions.mult_colwise(self.alpha_nk_gpu.gpudata, self.normalization_n_gpu.gpudata, self.N, self.K)
    return self.alpha_nk_gpu

  def calc_forces(self, alpha_nk_gpu):
    GPUFunctions.calc_forces(self.force_lambda, alpha_nk_gpu.gpudata, self.cloud_xyz_gpu.gpudata, self.model_xyz_gpu.gpudata, self.N, self.K, self.force_cache_nkd_gpu.gpudata)
    GPUFunctions.sum_forces_across_cloud(self.force_cache_nkd_gpu.gpudata, self.N, self.K, self.force_kd_gpu.gpudata)
    force_kd = self.force_kd_gpu.get().astype(float)
    return force_kd
