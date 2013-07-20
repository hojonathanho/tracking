import numpy as np
import trackingpy

def normalized(a):
  return a / np.linalg.norm(a)

class Cloth(object):
  def __init__(self, res_x, res_y, len_x, len_y, init_center):
    self.res_x, self.res_y, self.len_x, self.len_y, self.init_center = res_x, res_y, len_x, len_y, init_center

    # initialize node positions
    assert self.res_x >= 2 and self.res_y >= 2
    self.num_nodes = self.res_x * self.res_y

    self.init_pos = np.zeros((self.num_nodes, 3))
    self.init_pos[:,:2] = np.dstack(np.meshgrid(np.linspace(0, self.len_x, self.res_x), np.linspace(0, self.len_y, self.res_y))).reshape((-1, 2))
    self.init_pos -= self.init_pos.mean(axis=0) - self.init_center
    masses = np.ones(self.num_nodes)

    # create mass-spring system
    sim_params = trackingpy.SimulationParams()
    sim_params.dt = .01
    sim_params.solver_iters = 10
    sim_params.gravity = np.array([0, 0, -1.])
    sim_params.damping = 1
    sim_params.stretching_stiffness = 1
    sim_params.bending_stiffness = 1

    self.sys = trackingpy.MassSystem(self.init_pos, masses, sim_params)

    self._add_distance_constraints()
    self._declare_triangles_and_bending_constraints()
    self.sys.randomize_constraints()

  def _add_distance_constraints(self):
    self.distance_constraints = []
    def add_single(i, j):
      self.sys.add_distance_constraint(i, j, np.linalg.norm(self.init_pos[i] - self.init_pos[j]))
      self.distance_constraints.append((i, j))

    for i in range(self.num_nodes):
      x, y = self._i_to_xy(i)

      if x+1 < self.res_x:
        add_single(i, self._xy_to_i(x+1,y))

      if y+1 < self.res_y:
        add_single(i, self._xy_to_i(x,y+1))

      if x+1 < self.res_x and y+1 < self.res_y:
        add_single(i, self._xy_to_i(x+1,y+1))

      if x+1 < self.res_x and y-1 >= 0:
        add_single(i, self._xy_to_i(x+1,y-1))

  def _declare_triangles_and_bending_constraints(self):
    num_triangles = 2 * (self.res_x-1) * (self.res_y-1)
    self.triangles = np.empty((num_triangles, 3), dtype='int32')
    curr = 0
    for i in range(self.num_nodes):
      x, y = self._i_to_xy(i)
      if x+1 < self.res_x and y+1 < self.res_y:
        # switch up which way to draw the triangle hypotenuse
        # to break symmetries in the bending constraints
        if (curr / 2) % 2 == 0:
          self.triangles[curr,:]  = [i, self._xy_to_i(x+1,y), self._xy_to_i(x,y+1)]
          self.triangles[curr+1,:] = [self._xy_to_i(x+1,y), self._xy_to_i(x+1,y+1), self._xy_to_i(x,y+1)]
        else:
          self.triangles[curr,:]  = [i, self._xy_to_i(x+1,y+1), self._xy_to_i(x,y+1)]
          self.triangles[curr+1,:] = [i, self._xy_to_i(x+1,y), self._xy_to_i(x+1,y+1)]
        curr += 2
    assert curr == num_triangles
    self.sys.declare_triangles(self.triangles)

    # add bending constraints between all adjacent triangles
    # TODO: make this more efficient
    num_bending_constraints = 0
    for i, tri1 in enumerate(self.triangles):
      for tri2 in self.triangles[i+1:]:
        common_nodes = np.intersect1d(tri1, tri2)
        if len(common_nodes) != 2: continue
        p1, p2, p3, p4 = common_nodes[0], common_nodes[1], np.setdiff1d(tri1, common_nodes)[0], np.setdiff1d(tri2, common_nodes)[0]
        n1 = normalized(np.cross(self.init_pos[p2] - self.init_pos[p1], self.init_pos[p3] - self.init_pos[p1]))
        n2 = normalized(np.cross(self.init_pos[p2] - self.init_pos[p1], self.init_pos[p4] - self.init_pos[p1]))
        self.sys.add_bending_constraint(p1, p2, p3, p4, np.arccos(n1.dot(n2)))
        num_bending_constraints += 1
    print 'added', num_bending_constraints, 'bending constraints'


  def _i_to_xy(self, i): return i % self.res_x, i // self.res_x
  def _xy_to_i(self, x, y): return y*self.res_x + x

  def step(self):
    self.sys.step()

  def get_node_positions(self):
    return self.sys.get_node_positions()

  def get_distance_constraints(self):
    return self.distance_constraints
