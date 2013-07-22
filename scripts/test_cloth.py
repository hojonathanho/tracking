import numpy as np
import trackingpy
from trackingpy.cloth import Cloth

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

def main():
  import openravepy
  import trajoptpy
  env = openravepy.Environment()
  viewer = trajoptpy.GetViewer(env)

  env.LoadData(make_table_xml(translation=[0, 0, -.05], extents=[1, 1, .05]))

  np.random.seed(0)
  cloth = Cloth(res_x=10, res_y=20, len_x=.5, len_y=1., init_center=np.array([0, 0, .5]))

  # anchors for testing
  for i in range(cloth.res_x):
    cloth.sys.add_anchor_constraint(i, cloth.init_pos[i])

  # add above-table constraints
  for i in range(cloth.num_nodes):
    cloth.sys.add_plane_constraint(i, np.array([0, 0, 0]), np.array([0, 0, 1]))

  # from timeit import Timer
  # print 'timing'
  # num_timing_iters = 1000
  # t = Timer(lambda: cloth.step())
  # result = t.timeit(number=num_timing_iters)
  # print 'iters/sec =', float(num_timing_iters)/result, 'total time =', result
  # raw_input('done timing')

  iters = 100000

  log = np.empty((iters, cloth.num_nodes, 3))

  print 'cloth made'
  for i in range(iters):
    print i
    cloth.step()
    pos = cloth.get_node_positions()
    handles = [env.plot3(pos, 5)]
    handles.append(env.drawlinelist(pos[cloth.get_edges()].reshape((-1, 3)), 1, (0,1,0)))
    viewer.Idle()

    log[i,:,:] = pos

  # import cPickle
  # with open('out.pkl', 'w') as f:
  #   cPickle.dump(log, f)


if __name__ == '__main__':
  main()
