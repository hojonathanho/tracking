import numpy as np
import trackingpy
from trackingpy.cloth import Cloth

import openravepy, trajoptpy, cloudprocpy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
args = parser.parse_args()

def make_table_xml(translation, extents):
  xml = """
<Environment>
<KinBody name="table">
  <Body type="static" name="table_link">
    <Geom type="box">
      <Translation>%f %f %f</Translation>
      <extents>%f %f %f</extents>
      <diffuseColor>.96 .87 .70</diffuseColor>
    </Geom>
  </Body>
</KinBody>
</Environment>
""" % (translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
  return xml


def extract_red(rgb, depth, T_w_k):
  """
  extract red points and downsample
  """
  import cv2

  cx = 320.-.5
  cy = 240.-.5
  def depth_to_xyz(depth,f):
      x,y = np.meshgrid(np.arange(640), np.arange(480))
      assert depth.shape == (480, 640)
      XYZ = np.empty((480,640,3))
      Z = XYZ[:,:,2] = depth / 1000. # convert mm -> meters
      XYZ[:,:,0] = (x - cx)*(Z/f)
      XYZ[:,:,1] = (y - cy)*(Z/f)
      return XYZ
      
  def downsample(xyz, v):
      import cloudprocpy
      cloud = cloudprocpy.CloudXYZ()
      xyz1 = np.ones((len(xyz),4),'float')
      xyz1[:,:3] = xyz
      cloud.from2dArray(xyz1)
      cloud = cloudprocpy.downsampleCloud(cloud, v)
      return cloud.to2dArray()[:,:3]

  # T_h_k = np.array(
  #   [[-0.02102462, -0.03347223,  0.99921848, -0.186996  ],
  #    [-0.99974787, -0.00717795, -0.02127621,  0.04361884],
  #    [ 0.0078845,  -0.99941387, -0.03331288,  0.22145804],
  #    [ 0.,          0.,          0.,          1.        ]])
  # T_w_k = T_h_k

  hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
  h = hsv[:,:,0]
  s = hsv[:,:,1]
  v = hsv[:,:,2]

  red_mask = ((h<10) | (h>150)) & (s > 100) & (v > 100)

  valid_mask = depth > 0

  xyz_k = depth_to_xyz(depth, 544.260779961)
  xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]

  z = xyz_w[:,:,2]   
  z0 = xyz_k[:,:,2]
  height_mask = xyz_w[:,:,2] > .7 # TODO pass in parameter

  good_mask = red_mask & valid_mask
  good_xyz = xyz_w[good_mask]

  return downsample(good_xyz, .01)


def make_cloth_from_cloud(xyz):
  center = xyz.mean(axis=0)
  U, S, V = np.linalg.svd(xyz - center, full_matrices=False)
  # rotate so that smallest component is the z-direction
  T_w_k = np.eye(4)
  T_w_k[:3,:3] = V.T
  T_w_k[:3,3] = -center
  return T_w_k
  #return cloth, T_w_k

def draw_ax(T, size, env, handles):
    p0 = T[:3,3]
    xax, yax, zax = T[:3,:3].T*size
    width = size/10.
    handles.append(env.drawarrow(p0, p0+xax, width, [1,0,0]))
    handles.append(env.drawarrow(p0, p0+yax, width, [0,1,0]))
    handles.append(env.drawarrow(p0, p0+zax, width, [0,0,1]))

def main():
  env = openravepy.Environment()
  viewer = trajoptpy.GetViewer(env)

  # grabber = cloudprocpy.CloudGrabber()
  # grabber.startRGBD()

  print 'Reading input...'
  import h5py
  h5 = h5py.File(args.input, 'r')
  skip = 10
  rgbs, depths, T_w_k = h5['rgb'], h5['depth'], np.array(h5['T_w_k'])
  print 'done.'

  print 'Initializing'

  num_frames = len(rgbs)
  for i_frame in range(0, num_frames, skip):
    print 'Processing frame %d/%d' % (i_frame+1, num_frames)
    rgb, depth = rgbs[i_frame], depths[i_frame]
    # XYZ_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    # XYZ_w = XYZ_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]
    # handle = Globals.env.plot3(XYZ_w.reshape(-1,3), 2, rgb.reshape(-1,3)[:,::-1]/255.)
    xyz = extract_red(rgb, depth, T_w_k)
    handles = []
    handles.append(env.plot3(xyz, 2, (1,0,0)))
    draw_ax(T_w_k, .1, env, handles)

    # on the first frame, create cloth and table
    if i_frame == 0:
      table_height = xyz[:,2].mean() - .01
      env.LoadData(make_table_xml(translation=[0, 0, table_height-.05], extents=[1, 1, .05]))

      # calculate cloth dims, assuming laid out in x-y plane with no rotation
      cutoff_ind = int(len(xyz) * .01)
      argsort_x, argsort_y = xyz[:,0].argsort(), xyz[:,1].argsort()
      len_x = xyz[argsort_x[len(xyz)-cutoff_ind-1],0] - xyz[argsort_x[cutoff_ind],0]
      len_y = xyz[argsort_y[len(xyz)-cutoff_ind-1],1] - xyz[argsort_y[cutoff_ind],1]

      cloth = Cloth(res_x=10, res_y=15, len_x=len_x, len_y=len_y, init_center=xyz.mean(axis=0)+[0,0,.01])
      # add above-table constraints
      for i in range(cloth.num_nodes):
        cloth.sys.add_plane_constraint(i, np.array([0, 0, table_height]), np.array([0, 0, 1]))
      cloth_colors = np.zeros((cloth.num_nodes, 3))

    cloth.step()
  
    visible = np.asarray(cloth.sys.triangle_ray_test_against_nodes(T_w_k[:3,3])) == -1
    print visible, np.count_nonzero(visible)

    cloth_colors[:,:] = [0,0,1]
    cloth_colors[visible,:] = [1,1,1]
    handles.append(env.plot3(cloth.get_node_positions(), 5, cloth_colors))

    viewer.Idle()


  raw_input('asdfasdfasdf')
  return

  make_cloth_from_cloud(xyz)

  num_frames = len(rgbs)
  for i_frame in range(num_frames):
    rgb, depth = rgbs[i], depths[i]

  while True:
    # rgb, depth = grabber.getRGBD()
    # xyz = extract_red(rgb, depth)

    h = None
    if len(xyz) > 0:



      T_w_k = make_cloth_from_cloud(xyz)
      print T_w_k
      xyz = xyz.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,:]



      h = env.plot3(xyz, 5)
    viewer.Step()











  np.random.seed(0)
  cloth = Cloth(res_x=10, res_y=15, len_x=.5, len_y=1., init_center=np.array([0, 0, 0]))
  # add above-table constraints
  for i in range(cloth.num_nodes):
    cloth.sys.add_plane_constraint(i, np.array([0, 0, 0]), np.array([0, 0, 1]))

  i = 0
  while True:
    print i
    cloth.step()
    pos = cloth.get_node_positions()
    handles = [env.plot3(pos, 5)]
    for node_i, node_j in cloth.get_distance_constraints():
      handles.append(env.drawlinelist(np.asarray([pos[node_i], pos[node_j]]), 1, (0,1,0)))
    viewer.Step()
    i += 1



  h5.close()

if __name__ == '__main__':
  main()
