import numpy as np
import trackingpy
from trackingpy.cloth import Cloth
from trackingpy import tracking, clouds

import openravepy, trajoptpy, cloudprocpy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None)
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

def draw_ax(T, size, env, handles):
    p0 = T[:3,3]
    xax, yax, zax = T[:3,:3].T*size
    width = size/10.
    handles.append(env.drawarrow(p0, p0+xax, width, [1,0,0]))
    handles.append(env.drawarrow(p0, p0+yax, width, [0,1,0]))
    handles.append(env.drawarrow(p0, p0+zax, width, [0,0,1]))

def norms(x, axis=0):
    return np.sqrt((x**2).sum(axis=axis))

def initialize_cloth(env, cloud_xyz):
  table_height = cloud_xyz[:,2].mean() - .01
  env.LoadData(make_table_xml(translation=[0, 0, table_height-.05], extents=[1, 1, .05]))

  # calculate cloth dims, assuming laid out in x-y plane with no rotation
  cutoff_ind = int(len(cloud_xyz) * .01)
  argsort_x, argsort_y = cloud_xyz[:,0].argsort(), cloud_xyz[:,1].argsort()
  len_x = cloud_xyz[argsort_x[len(cloud_xyz)-cutoff_ind-1],0] - cloud_xyz[argsort_x[cutoff_ind],0]
  len_y = cloud_xyz[argsort_y[len(cloud_xyz)-cutoff_ind-1],1] - cloud_xyz[argsort_y[cutoff_ind],1]


  sim_params = trackingpy.SimulationParams()
  sim_params.dt = .01
  sim_params.solver_iters = 10
  sim_params.gravity = np.array([0, 0, -1.])
  sim_params.damping = 10
  sim_params.stretching_stiffness = 1
  sim_params.bending_stiffness = .7

  cloth = Cloth(res_x=10, res_y=30, len_x=len_x, len_y=len_y, init_center=cloud_xyz.mean(axis=0)+[0,0,.01], total_mass=300, sim_params=sim_params)
  # add above-table constraints
  for i in range(cloth.num_nodes):
    cloth.sys.add_plane_constraint(i, np.array([0, 0, table_height]), np.array([0, 0, 1]))

  return cloth


def plot(env, handles, cloud_xyz, T_w_k, cloth, occluded_in_depth_img=None, occluded_by_model=None, force_kd=None):
  handles.append(env.plot3(cloud_xyz, 2, (1,0,0)))
  draw_ax(T_w_k, .1, env, handles)
  pos = cloth.get_node_positions()

  cloth_colors = np.ones((cloth.num_nodes, 3), dtype=float)
  if occluded_by_model is not None and occluded_in_depth_img is not None:
    cloth_colors[occluded_in_depth_img,:] = [0,0,1]
    cloth_colors[occluded_by_model,:] = [0,0,0]
  handles.append(env.plot3(pos, 10, cloth_colors))
  handles.append(env.drawlinelist(pos[cloth.get_edges()].reshape((-1, 3)), 1, (0,1,0)))

  if force_kd is not None:
    for i in range(cloth.num_nodes):
      handles.append(env.drawarrow(pos[i], pos[i]+.05*(force_kd[i]/10), .0005))

def main():
  pnoise = .01
  pinvisible = .1
  sigma = np.eye(3) * .0001
  force_lambda = 100
  num_em_iters = 5
  depth_occlude_tol = .03

  env = openravepy.Environment()
  viewer = trajoptpy.GetViewer(env)

  if args.input is None:
    live = True
    print 'No input file specified, streaming clouds live'
    def stream_rgbd():
      import rospy
      from rapprentice import berkeley_pr2, PR2
      rospy.init_node("tracker", disable_signals=True)
      pr2 = PR2.PR2()
      grabber = cloudprocpy.CloudGrabber()
      grabber.startRGBD()
      while True:
        print 'Grabbing cloud...'
        rgb, depth = grabber.getRGBD()
        pr2.update_rave()
        T_w_k = berkeley_pr2.get_kinect_transform(pr2.robot)
        print 'done'
        yield rgb, depth, T_w_k

  else:
    live = False
    print 'Reading input file', args.input
    import h5py
    def stream_rgbd():
      h5 = h5py.File(args.input, 'r')
      rgbs, depths, T_w_k = h5['rgb'], h5['depth'], np.array(h5['T_w_k'])
      num_frames = len(rgbs)
      skip = 2
      start = 160
      for i_frame in range(start, num_frames, skip):
        #print 'Processing frame %d/%d. Number of points in cloud: %d' % (i_frame+1, num_frames, len(cloud_xyz))
        print 'Processing frame %d/%d.' % (i_frame+1, num_frames)
        yield rgbs[i_frame], depths[i_frame], T_w_k


  for i_frame, (rgb, depth, T_w_k) in enumerate(stream_rgbd()):
    cloud_xyz = clouds.extract_color(rgb, depth, T_w_k)
    if len(cloud_xyz) == 0:
      print 'Filtered cloud is empty. Skipping frame.'
      continue

    # Initialization: on the first frame, create cloth and table
    if i_frame == 0:
      cloth = initialize_cloth(env, cloud_xyz)
      print 'Initialization ok?'
      handles = []; plot(env, handles, cloud_xyz, T_w_k, cloth)
      viewer.Idle()

    for em_iter in range(num_em_iters):
      print 'EM iteration %d/%d' % (em_iter+1, num_em_iters)
      model_xyz = cloth.get_node_positions()

      # Calculate visibility
      # check for nodes occluded by other nodes by raycasting from the camera
      raytest_results = np.asarray(cloth.sys.triangle_ray_test_against_nodes(T_w_k[:3,3]))
      occluded_by_model = raytest_results >= 0
      visibility = np.ones_like(occluded_by_model, dtype=float)
      visibility[occluded_by_model] = pinvisible
      # check if depth image contains a point significantly in front of a model node
      dist_cam2node = norms(T_w_k[:3,3][None,:] - model_xyz, axis=1)
      T_k_w = np.linalg.inv(T_w_k)
      depth_cam2node = clouds.lookup_depth_by_xyz(depth, model_xyz.dot(T_k_w[:3,:3].T) + T_k_w[:3,3])
      occluded_in_depth_img = depth_cam2node < (dist_cam2node - depth_occlude_tol)
      visibility[occluded_in_depth_img] = pinvisible

      # Calculate expected correspondences
      alpha_NK = tracking.mvn_densities(cloud_xyz, model_xyz, sigma) * visibility[None,:]
      alpha_NK /= (alpha_NK.sum(axis=1) + pnoise)[:,None]

      # Calculate and apply forces
      force_kd = force_lambda * (alpha_NK[:,:,None] * (cloud_xyz[:,None,:] - model_xyz[None,:,:])).sum(axis=0)
      cloth.sys.apply_forces(force_kd)
      cloth.step()

    handles = []; plot(env, handles, cloud_xyz, T_w_k, cloth, occluded_in_depth_img, occluded_by_model, force_kd)

    if live:
      viewer.Step()
    else:
      viewer.Idle()


if __name__ == '__main__':
  main()
