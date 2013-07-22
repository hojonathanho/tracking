import numpy as np

def mvn_densities(x_nd, m_kd, cov_dd):
  """
  Given an array of N D-dimensional points, K means, and a single covariance matrix,
    returns a NxK array of Gaussian densities
  """

  n, k, d = x_nd.shape[0], m_kd.shape[0], m_kd.shape[1]
  assert x_nd.shape[1] == d and cov_dd.shape == (d, d)

  cov_det, cov_inv = np.linalg.det(cov_dd), np.linalg.inv(cov_dd)

  diffs = x_nd[:,None,:] - m_kd[None,:,:]
  diffs.shape = (-1, 3)

  out = np.exp(-.5 * (diffs.dot(cov_inv.T) * diffs).sum(axis=-1))
  out /= np.sqrt(np.power(2.*np.pi, d) * cov_det)
  out.shape = (n, k)

  return out


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

if __name__ == '__main__':
  unittest.main()
