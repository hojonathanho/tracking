import numpy as np
import trackingpy
from trackingpy.cloth import TriangleMesh, Cloth
from trackingpy import tracking, clouds

import openravepy, trajoptpy, cloudprocpy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--gpu', action='store_true')
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

def initialize_cloth(env, cloud_xyz):
  table_height = cloud_xyz[:,2].mean() - .01
  env.LoadData(make_table_xml(translation=[0, 0, table_height-.05], extents=[1, 1, .05]))

  sim_params = trackingpy.SimulationParams()
  sim_params.dt = .01
  sim_params.solver_iters = 10
  sim_params.gravity = np.array([0, 0, -1.])
  sim_params.damping = 10
  sim_params.stretching_stiffness = 1
  sim_params.bending_stiffness = .7
  # return TriangleMesh.FromObjFile('/home/jonathan/Desktop/simple_cloth_with_slit.obj', init_center=cloud_xyz.mean(axis=0)+[0,0,.01], total_mass=300, sim_params=sim_params)

  # calculate cloth dims, assuming laid out in x-y plane with no rotation
  cutoff_ind = int(len(cloud_xyz) * .01)
  argsort_x, argsort_y = cloud_xyz[:,0].argsort(), cloud_xyz[:,1].argsort()
  len_x = cloud_xyz[argsort_x[len(cloud_xyz)-cutoff_ind-1],0] - cloud_xyz[argsort_x[cutoff_ind],0]
  len_y = cloud_xyz[argsort_y[len(cloud_xyz)-cutoff_ind-1],1] - cloud_xyz[argsort_y[cutoff_ind],1]
  # len_x = .6
  # len_y = .52
  res_x = 10#20
  res_y = 30#22

  cloth = Cloth(res_x=res_x, res_y=res_y, len_x=len_x, len_y=len_y, init_center=cloud_xyz.mean(axis=0)+[0,0,.01], total_mass=300, sim_params=sim_params)
  # with open('/home/jonathan/Desktop/simple_cloth.obj', 'w') as f: f.write(cloth.dump_obj_data())
  # add above-table constraints
  for i in range(cloth.get_num_nodes()):
    cloth.sys.add_plane_constraint(i, np.array([0, 0, table_height]), np.array([0, 0, 1]))
  return cloth


def plot(env, handles, cloud_xyz, T_w_k, cloth, occluded_in_depth_img=None, occluded_by_model=None, force_kd=None):
  handles.append(env.plot3(cloud_xyz, 5, (1,0,0)))
  draw_ax(T_w_k, .1, env, handles)
  pos = cloth.get_node_positions()

  cloth_colors = np.ones((cloth.get_num_nodes(), 3), dtype=float)
  if occluded_by_model is not None and occluded_in_depth_img is not None:
    cloth_colors[occluded_in_depth_img,:] = [0,0,1]
    cloth_colors[occluded_by_model,:] = [0,0,0]
  handles.append(env.plot3(pos, 10, cloth_colors))
  handles.append(env.drawlinelist(cloth.get_edge_positions().reshape((-1, 3)), 1, (0,1,0)))

  if force_kd is not None:
    for i in range(cloth.get_num_nodes()):
      handles.append(env.drawarrow(pos[i], pos[i]+.05*(force_kd[i]/10), .0005))

def main():

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
        print 'Processing frame %d/%d.' % (i_frame+1, num_frames)
        yield rgbs[i_frame], depths[i_frame], T_w_k


  for i_frame, (rgb, depth, T_w_k) in enumerate(stream_rgbd()):
    cloud_xyz = clouds.extract_color(rgb, depth, T_w_k, clouds.red_mask)
    if len(cloud_xyz) == 0:
      print 'Filtered cloud is empty. Skipping frame.'
      continue

    # Initialization: on the first frame, create cloth and table
    if i_frame == 0:
      cloth = initialize_cloth(env, cloud_xyz)
      print 'Initialization ok?'
      handles = []; plot(env, handles, cloud_xyz, T_w_k, cloth)
      viewer.Idle()
      if args.gpu:
        from trackingpy import tracking_gpu
        tracker = tracking_gpu.GPUTracker(cloth)
      else:
        tracker = tracking.Tracker(cloth)

    tracker.set_input(cloud_xyz, depth, T_w_k)
    out = tracker.step(return_data=True)
    handles = []; plot(env, handles, cloud_xyz, T_w_k, cloth, out['occluded_in_depth_img'], out['occluded_by_model'], out['force_kd'])

    if live:
      viewer.Step()
    else:
      viewer.Idle()


if __name__ == '__main__':
  main()
