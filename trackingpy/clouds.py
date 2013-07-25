import cloudprocpy
import cv2
import numpy as np

class DepthCameraParams(object):
  nx = 640
  ny = 480
  cx = 320. - .5
  cy = 240. - .5
  f = 544.260779961

def depth_to_xyz(depth, depth_scale):
  x,y = np.meshgrid(np.arange(DepthCameraParams.nx), np.arange(DepthCameraParams.ny))
  assert depth.shape == (DepthCameraParams.ny, DepthCameraParams.nx)
  XYZ = np.empty((DepthCameraParams.ny, DepthCameraParams.nx, 3))
  Z = XYZ[:,:,2] = depth * depth_scale
  XYZ[:,:,0] = (x - DepthCameraParams.cx)*(Z/DepthCameraParams.f)
  XYZ[:,:,1] = (y - DepthCameraParams.cy)*(Z/DepthCameraParams.f)
  return XYZ

def lookup_depth_by_xyz(depth, xyz):
  assert len(xyz.shape) == 2 and xyz.shape[1] == 3
  x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]

  img_x = np.rint(x/z*DepthCameraParams.f + DepthCameraParams.cx).astype(int)
  img_y = np.rint(y/z*DepthCameraParams.f + DepthCameraParams.cy).astype(int)
  oob_x = (img_x < 0) | (img_x >= depth.shape[1])
  oob_y = (img_y < 0) | (img_y >= depth.shape[0])
  img_x[oob_x] = 0
  img_y[oob_y] = 0

  out = depth[img_y, img_x]
  out[oob_x | oob_y] = -1
  return out

def downsample(xyz, v):
  cloud = cloudprocpy.CloudXYZ()
  xyz1 = np.ones((len(xyz),4),'float')
  xyz1[:,:3] = xyz
  cloud.from2dArray(xyz1)
  cloud = cloudprocpy.downsampleCloud(cloud, v)
  return cloud.to2dArray()[:,:3]

def everything_mask(h, s, v):
  return np.ones_like(h, dtype=bool)

def red_mask(h, s, v):
  return ((h < 10) | (h > 150)) & (s > 100) & (v > 100)

def yellow_mask(h, s, v):
  return (h > 20) & (h < 30) & (s > 10) & (v > 100)

def extract_color(rgb, depth, T_w_k, color_mask_func=red_mask, min_height=.7, ds=.015):
  """
  extract red points and downsample
  """

  hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
  h = hsv[:,:,0]
  s = hsv[:,:,1]
  v = hsv[:,:,2]
  color_mask = color_mask_func(h, s, v)

  valid_mask = depth > 0

  xyz_k = depth_to_xyz(depth, depth_scale=1/1000.) # depth_scale converts mm -> meters
  xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]

  z = xyz_w[:,:,2]   
  z0 = xyz_k[:,:,2]
  height_mask = xyz_w[:,:,2] > min_height

  good_mask = color_mask & valid_mask
  good_xyz = xyz_w[good_mask]

  return downsample(good_xyz, ds)

import unittest
class Tests(unittest.TestCase):
  def test_lookup_depth_by_xyz(self):
    depth = np.random.rand(480, 640)
    xyz = depth_to_xyz(depth, depth_scale=1)
    self.assertTrue(np.allclose(depth, lookup_depth_by_xyz(depth, xyz.reshape((-1,3))).reshape(480, 640)))

if __name__ == '__main__':
  unittest.main()
