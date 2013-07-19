import openravepy, trajoptpy, cloudprocpy
import numpy as np
import h5py
import argparse
import subprocess
from rapprentice import berkeley_pr2, PR2, clouds
import cv2

cmap = np.zeros((256, 3),dtype='uint8')
cmap[:,0] = range(256)
cmap[:,2] = range(256)[::-1]
cmap[0] = [0,0,0]

class Globals(object):
  pr2 = None
  env = None
  robot = None


def extract_red(rgb, depth, T_w_k):
    """
    extract red points and downsample
    """
        
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    red_mask = ((h<10) | (h>150)) & (s > 100) & (v > 100)
    
    valid_mask = depth > 0
    
    xyz_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]
    
    z = xyz_w[:,:,2]   
    z0 = xyz_k[:,:,2]
    height_mask = xyz_w[:,:,2] > .7 # TODO pass in parameter
    
    
    good_mask = red_mask & height_mask & valid_mask

    good_xyz = xyz_w[good_mask]
    

    return clouds.downsample(good_xyz, .01)

def draw_ax(T, size, env, handles):
    p0 = T[:3,3]
    xax, yax, zax = T[:3,:3].T*size
    width = size/10.
    handles.append(env.drawarrow(p0, p0+xax, width, [1,0,0]))
    handles.append(env.drawarrow(p0, p0+yax, width, [0,1,0]))
    handles.append(env.drawarrow(p0, p0+zax, width, [0,0,1]))

def main():
  subprocess.call("killall XnSensorServer", shell=True)

  parser = argparse.ArgumentParser()
  parser.add_argument('file', type=str)
  parser.add_argument('--real_robot', action='store_true')
  parser.add_argument('--view', action='store_true')
  args = parser.parse_args()

  if args.view:
    print 'Opening file %s' % args.file
    f = h5py.File(args.file, 'r')
    rgbs = f['rgb']
    depths = f['depth']
    T_w_k = np.array(f['T_w_k'])
    Globals.env = openravepy.Environment()
    viewer = trajoptpy.GetViewer(Globals.env)

    num_frames = len(rgbs)
    for i in range(0, num_frames, 10):
      print 'Showing frame %d/%d' % (i+1, num_frames)
      rgb, depth = rgbs[i], depths[i]
      # XYZ_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
      # XYZ_w = XYZ_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]
      # handle = Globals.env.plot3(XYZ_w.reshape(-1,3), 2, rgb.reshape(-1,3)[:,::-1]/255.)
      xyz = extract_red(rgb, depth, T_w_k)
      handles = []
      handles.append(Globals.env.plot3(xyz, 2, (1,0,0)))
      draw_ax(T_w_k, .1, Globals.env, handles)
      viewer.Idle()

    return

  if args.real_robot:
    import rospy
    rospy.init_node("recorder", disable_signals=True)
    Globals.pr2 = PR2.PR2()
    Globals.env = Globals.pr2.env
    Globals.robot = Globals.pr2.robot

    Globals.pr2.update_rave()
    T_w_k = berkeley_pr2.get_kinect_transform(Globals.robot)
  else:
    Globals.env = openravepy.Environment()
    T_w_k = np.eye(4)

  viewer = trajoptpy.GetViewer(Globals.env)
  #env.LoadData(make_table_xml(translation=[0, 0, -.05], extents=[1, 1, .05]))

  num_frames = 0
  rgbs, depths = [], []

  grabber = cloudprocpy.CloudGrabber()
  grabber.startRGBD()
  try:
    while True:
      rgb, depth = grabber.getRGBD()
      rgbs.append(rgb)
      depths.append(depth)

      cv2.imshow("rgb", rgb)
      cv2.imshow("depth", cmap[np.fmin((depth*.064).astype('int'), 255)])
      cv2.waitKey(30)

      num_frames += 1
      print 'Captured frame', num_frames
      viewer.Step()

  except KeyboardInterrupt:
    print 'Keyboard interrupt'

  print 'Writing to %s ...' % args.file
  out = h5py.File(args.file, 'w')
  out.create_dataset('rgb', data=rgbs, compression='gzip', chunks=(1, 256, 256, 3))
  out.create_dataset('depth', data=depths, compression='gzip', chunks=(1, 256, 256))
  out['T_w_k'] = T_w_k
  out.close()


if __name__ == '__main__':
  main()
