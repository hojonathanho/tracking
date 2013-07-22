from collections import defaultdict
import numpy as np
import trackingpy

class Cloth(object):
  def __init__(self, res_x, res_y, len_x, len_y, init_center, total_mass, sim_params):
    self.res_x, self.res_y, self.len_x, self.len_y, self.init_center = res_x, res_y, len_x, len_y, init_center

    # initialize node positions
    assert self.res_x >= 2 and self.res_y >= 2
    self.num_nodes = self.res_x * self.res_y

    self.init_pos = np.zeros((self.num_nodes, 3))
    self.init_pos[:,:2] = np.dstack(np.meshgrid(np.linspace(0, self.len_x, self.res_x), np.linspace(0, self.len_y, self.res_y))).reshape((-1, 2))
    self.init_pos -= self.init_pos.mean(axis=0) - self.init_center
    masses = np.ones(self.num_nodes) * (float(total_mass) / self.num_nodes)
    print masses

    # create mass-spring system
    self.sys = trackingpy.MassSystem(self.init_pos, masses, sim_params)

    # declare which triples of nodes constitute triangles
    # (currently only used for raycasting)
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

    # add distance constraints for all edges
    edge2triangles = defaultdict(list) # map of edge -> list of triangles
    for tri in self.triangles:
      edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
      for edge_i, edge_j in edges:
        i, j = min(edge_i, edge_j), max(edge_i, edge_j)
        edge2triangles[(i,j)].append(tri)
        if len(edge2triangles[(i,j)]) > 1: continue
        self.sys.add_distance_constraint(i, j, np.linalg.norm(self.init_pos[i] - self.init_pos[j]))
    self.edges = np.array(edge2triangles.keys())

    # add bending constraints between all pairs of adjacent triangles
    num_bending_constraints = 0
    for triangles in edge2triangles.itervalues():
      if len(triangles) == 1: continue # this edge lies on the cloth border
      assert len(triangles) == 2 # non-border edges must be a part of two triangles
      tri1, tri2 = triangles
      common_nodes = np.intersect1d(tri1, tri2)
      assert len(common_nodes) == 2
      p1, p2, p3, p4 = common_nodes[0], common_nodes[1], np.setdiff1d(tri1, common_nodes)[0], np.setdiff1d(tri2, common_nodes)[0]
      n1 = np.cross(self.init_pos[p2] - self.init_pos[p1], self.init_pos[p3] - self.init_pos[p1]); n1 /= np.linalg.norm(n1)
      n2 = np.cross(self.init_pos[p2] - self.init_pos[p1], self.init_pos[p4] - self.init_pos[p1]); n2 /= np.linalg.norm(n2)
      self.sys.add_bending_constraint(p1, p2, p3, p4, np.arccos(n1.dot(n2)))
      num_bending_constraints += 1

    self.sys.randomize_constraints()


  def _i_to_xy(self, i): return i % self.res_x, i // self.res_x
  def _xy_to_i(self, x, y): return y*self.res_x + x

  def step(self):
    self.sys.step()

  def get_node_positions(self):
    return self.sys.get_node_positions()

  def get_edges(self):
    return self.edges
