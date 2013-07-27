import numpy as np
import trackingpy
from trackingpy.model import TriangleMesh, Cloth
from trackingpy import tracking, clouds, miniviewer

import cloudprocpy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None, help='Optionally provide a recording (from recorder.py). If not provided, point clouds will be streamed live.')
parser.add_argument('--gpu', action='store_true', help='Use the GPU (CUDA) implementation.')
parser.add_argument('--obj_type', choices=['cloth', 'suture_pad'], default='cloth')
args = parser.parse_args()


def plot(viewer, handles, cloud_xyz, T_w_k, obj, occluded_in_depth_img=None, occluded_by_model=None, force_kd=None):
  handles.append(viewer.plot3(cloud_xyz, 2, (1,0,0)))
  handles.append(viewer.drawaxes(T_w_k, .1))
  pos = obj.get_node_positions()

  obj_colors = np.ones((obj.get_num_nodes(), 3), dtype=float)
  if occluded_by_model is not None and occluded_in_depth_img is not None:
    obj_colors[occluded_in_depth_img,:] = [0,0,1]
    obj_colors[occluded_by_model,:] = [0,0,0]
  handles.append(viewer.plot3(pos, 10, obj_colors))
  handles.append(viewer.drawlinelist(obj.get_edge_positions().reshape((-1, 3)), .1, (0,0,0)))

  # if force_kd is not None:
  #   for i in range(obj.get_num_nodes()):
  #     handles.append(viewer.drawarrow(pos[i], pos[i]+.05*(force_kd[i]/10), .0005))

  viewer.block_until_all_drawn()

def initialize_tracked_obj(viewer, handles, cloud_xyz):
  sim_params = trackingpy.SimulationParams()
  sim_params.dt = .01
  sim_params.solver_iters = 10
  sim_params.gravity = np.array([0, 0, -1.])
  sim_params.damping = 10
  sim_params.stretching_stiffness = 1
  sim_params.bending_stiffness = .7

  if args.obj_type == 'suture_pad':
    import os
    filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(trackingpy.__file__))), 'data', 'simple_cloth_with_slit.obj')
    return TriangleMesh.FromObjFile(filename, init_center=cloud_xyz.mean(axis=0)+[0,0,.01], total_mass=300, sim_params=sim_params)

  elif args.obj_type == 'cloth':
    # draw a table at a height just below the cloth
    table_height = cloud_xyz[:,2].mean() - .01
    handles.append(viewer.drawbox(center=np.array([0, 0, table_height-.05]), extents=np.array([1, 1, .05]), color=np.array([.96, .87, .70])))
    # calculate cloth dims, assuming laid out in x-y plane with no rotation
    cutoff_ind = int(len(cloud_xyz) * .01)
    argsort_x, argsort_y = cloud_xyz[:,0].argsort(), cloud_xyz[:,1].argsort()
    len_x = cloud_xyz[argsort_x[len(cloud_xyz)-cutoff_ind-1],0] - cloud_xyz[argsort_x[cutoff_ind],0]
    len_y = cloud_xyz[argsort_y[len(cloud_xyz)-cutoff_ind-1],1] - cloud_xyz[argsort_y[cutoff_ind],1]
    res_x = 10
    res_y = 30
    cloth = Cloth(res_x=res_x, res_y=res_y, len_x=len_x, len_y=len_y, init_center=cloud_xyz.mean(axis=0)+[0,0,.01], total_mass=300, sim_params=sim_params)
    # add above-table constraints
    for i in range(cloth.get_num_nodes()):
      cloth.sys.add_plane_constraint(i, np.array([0, 0, table_height]), np.array([0, 0, 1]))
    return cloth

  else:
    assert False


def main():
  viewer = miniviewer.make_viewer()

  # Define point cloud-reading stream, depending on input method
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


  # Tracking loop: run EM iterations for each point cloud received
  tracked_obj = None
  for rgb, depth, T_w_k in stream_rgbd():
    mask_func = clouds.yellow_mask if args.obj_type == 'suture_pad' else clouds.red_mask
    cloud_xyz = clouds.extract_color(rgb, depth, T_w_k, mask_func)
    if len(cloud_xyz) == 0:
      print 'Filtered cloud is empty. Skipping frame.'
      continue

    # Initialization: on the first frame, create cloth and table
    if tracked_obj is None:
      init_handles = []
      tracked_obj = initialize_tracked_obj(viewer, init_handles, cloud_xyz)
      print 'Initialization ok?'
      handles = []; plot(viewer, handles, cloud_xyz, T_w_k, tracked_obj)
      #viewer.Idle()
      if args.gpu:
        from trackingpy import tracking_gpu
        tracker = tracking_gpu.GPUTracker(tracked_obj)
      else:
        tracker = tracking.Tracker(tracked_obj)

    tracker.set_input(cloud_xyz, depth, T_w_k)
    out = tracker.step(return_data=True)
    handles = []; plot(viewer, handles, cloud_xyz, T_w_k, tracked_obj, out['occluded_in_depth_img'], out['occluded_by_model'], out['force_kd'])

    if not live:
      raw_input('Press enter for the next frame')

if __name__ == '__main__':
  main()
