import numpy as np
import trackingpy
from trackingpy.cloth import Cloth
from trackingpy import tracking

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


def depth_to_xyz(depth, f=544.260779961):
  cx = 320.-.5
  cy = 240.-.5
  x,y = np.meshgrid(np.arange(640), np.arange(480))
  assert depth.shape == (480, 640)
  XYZ = np.empty((480,640,3))
  Z = XYZ[:,:,2] = depth / 1000. # convert mm -> meters
  XYZ[:,:,0] = (x - cx)*(Z/f)
  XYZ[:,:,1] = (y - cy)*(Z/f)
  return XYZ

def extract_red(rgb, depth, T_w_k):
  """
  extract red points and downsample
  """
  import cv2

  def downsample(xyz, v):
    import cloudprocpy
    cloud = cloudprocpy.CloudXYZ()
    xyz1 = np.ones((len(xyz),4),'float')
    xyz1[:,:3] = xyz
    cloud.from2dArray(xyz1)
    cloud = cloudprocpy.downsampleCloud(cloud, v)
    return cloud.to2dArray()[:,:3]

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

  return downsample(good_xyz, .015), xyz_w

def points_to_depth(T_w_k, depth, points_w, f=544.260779961):
  cx = 320.-.5
  cy = 240.-.5
  T_k_w = np.linalg.inv(T_w_k)
  points_k = points_w.dot(T_k_w[:3,:3].T) + T_k_w[:3,3]
  x, y, z = points_k[:,0], points_k[:,1], points_k[:,2]
  img_x, img_y = (x*f/z + cx).astype(int), (y*f/z + cy).astype(int)
  max_x, max_y = depth.shape[0], depth.shape[1]
  oob_x = img_x >= max_x; oob_y = img_y >= max_y
  img_x[oob_x] = 0; img_y[oob_y] = 0
  out = depth[img_x, img_y]
  out[oob_x | oob_y] = -1
  return out


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

def norms(x, axis=0):
    return np.sqrt((x**2).sum(axis=axis))

def main():
  pnoise = .01
  pinvisible = .1
  sigma = np.eye(3) * .0001
  force_lambda = 100
  num_em_iters = 10
  depth_occlude_tol = .03


  env = openravepy.Environment()
  viewer = trajoptpy.GetViewer(env)

  # grabber = cloudprocpy.CloudGrabber()
  # grabber.startRGBD()

  print 'Reading input...'
  import h5py
  h5 = h5py.File(args.input, 'r')
  rgbs, depths, T_w_k = h5['rgb'], h5['depth'], np.array(h5['T_w_k'])
  print 'done.'

  print 'Initializing'

  num_frames = len(rgbs)
  skip = 2
  start = 160
  first = True
  for i_frame in range(start, num_frames, skip):
    # Read point cloud
    rgb, depth = rgbs[i_frame], depths[i_frame]
    # XYZ_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    # XYZ_w = XYZ_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]
    # handle = Globals.env.plot3(XYZ_w.reshape(-1,3), 2, rgb.reshape(-1,3)[:,::-1]/255.)
    cloud_xyz, xyz_w = extract_red(rgb, depth, T_w_k)
    print 'Processing frame %d/%d. Number of points in cloud: %d' % (i_frame+1, num_frames, len(cloud_xyz))

    # Initialization: on the first frame, create cloth and table
    if first:
      first = False
      table_height = cloud_xyz[:,2].mean() - .01
      env.LoadData(make_table_xml(translation=[0, 0, table_height-.05], extents=[1, 1, .05]))

      # calculate cloth dims, assuming laid out in x-y plane with no rotation
      cutoff_ind = int(len(cloud_xyz) * .01)
      argsort_x, argsort_y = cloud_xyz[:,0].argsort(), cloud_xyz[:,1].argsort()
      len_x = cloud_xyz[argsort_x[len(cloud_xyz)-cutoff_ind-1],0] - cloud_xyz[argsort_x[cutoff_ind],0]
      len_y = cloud_xyz[argsort_y[len(cloud_xyz)-cutoff_ind-1],1] - cloud_xyz[argsort_y[cutoff_ind],1]

      cloth = Cloth(res_x=10, res_y=30, len_x=len_x, len_y=len_y, init_center=cloud_xyz.mean(axis=0)+[0,0,.01])
      # add above-table constraints
      for i in range(cloth.num_nodes):
        cloth.sys.add_plane_constraint(i, np.array([0, 0, table_height]), np.array([0, 0, 1]))
      cloth_colors = np.zeros((cloth.num_nodes, 3))

      # avg distance between nodes

    for em_iter in range(num_em_iters):
      print 'EM iteration %d/%d' % (em_iter+1, num_em_iters)
      # Calculate visibility
      print 'camera pos', T_w_k[:3,3]
      raytest_results = np.asarray(cloth.sys.triangle_ray_test_against_nodes(T_w_k[:3,3]))
      occluded_by_model = raytest_results >= 0
      visibility = np.ones_like(occluded_by_model, dtype=float)
      visibility[occluded_by_model] = pinvisible
      # check if depth image contains a point significantly in front of the model node


      dist_cam2node = norms(T_w_k[:3,3][None,:] - cloth.sys.get_node_positions(), axis=1)
      depth_cam2node = points_to_depth(T_w_k, depth, cloth.sys.get_node_positions())
      occluded_in_depth_img = depth_cam2node < (dist_cam2node - depth_occlude_tol)
      visibility[occluded_in_depth_img] = pinvisible



      # Calculate expected correspondences
      model_xyz = cloth.get_node_positions()
      assert len(visibility) == len(model_xyz)
      densities_NK = tracking.mvn_densities(cloud_xyz, model_xyz, sigma)

      alpha_NK = densities_NK * visibility[None,:]
      alpha_NK /= (alpha_NK.sum(axis=1) + pnoise)[:,None]

      # Calculate and apply forces
      force_kd = force_lambda * (alpha_NK[:,:,None] * (cloud_xyz[:,None,:] - model_xyz[None,:,:])).sum(axis=0)
      cloth.sys.apply_forces(force_kd)
      print 'max force applied:', np.linalg.norm(force_kd.max(axis=0))
      cloth.step()

    # Plotting
    handles = []
    handles.append(env.plot3(cloud_xyz, 2, (1,0,0)))
    draw_ax(T_w_k, .1, env, handles)

    cloth_colors[raytest_results < 0,:] = [1,1,1]
    cloth_colors[occluded_in_depth_img,:] = [0,0,1]
    cloth_colors[occluded_by_model,:] = [0,0,0]
    pos = cloth.get_node_positions()
    handles.append(env.plot3(pos, 10, cloth_colors))
    handles.append(env.drawlinelist(pos[np.array(cloth.get_distance_constraints())].reshape((-1, 3)), 1, (0,1,0)))

    # raytest_results[not_visible]
    # for i in range(cloth.num_nodes):
    #   if not_visible[i]:
    #     targ = pos[cloth.triangles[raytest_results[i]]].mean(axis=0)
    #     handles.append(env.drawarrow(pos[i], targ, .001, [1,0,0]))

    for i in range(cloth.num_nodes):
      handles.append(env.drawarrow(pos[i], pos[i]+.05*(force_kd[i]/10), .0005))

    viewer.Idle()


if __name__ == '__main__':
  main()
