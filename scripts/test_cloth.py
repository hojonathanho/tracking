import numpy as np
import trackingpy
from trackingpy.model import Model, Mesh

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
  sim_params = trackingpy.SimulationParams()
  sim_params.dt = .01
  sim_params.solver_iters = 10
  sim_params.gravity = np.array([0, 0, -1.])
  sim_params.damping = 1
  sim_params.stretching_stiffness = 1
  sim_params.bending_stiffness = .7
  cloth = Mesh('/home/jonathan/Desktop/simple_cloth_with_slit.obj', 100, sim_params)
  # cloth = Cloth(res_x=50, res_y=50, len_x=.5, len_y=1., init_center=np.array([0, 0, .5]), total_mass=100, sim_params=sim_params)

  # anchors for testing
  # for i in range(cloth.res_x):
  #   cloth.sys.add_anchor_constraint(i, cloth.init_pos[i])

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

  print 'cloth made'
  while True:
    cloth.step()
    pos = cloth.get_node_positions()
    handles = [env.plot3(pos, 5)]
    handles.append(env.drawlinelist(cloth.get_edge_positions().reshape((-1, 3)), 1, (0,1,0)))
    viewer.Idle()

if __name__ == '__main__':
  main()
