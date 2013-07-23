from collections import defaultdict
import numpy as np
import trackingpy

def make_mass_system(init_pos, masses, triangles, sim_params):
  # create mass-spring system
  sys = trackingpy.MassSystem(init_pos, masses, sim_params)
  sys.declare_triangles(triangles)

  # add distance constraints for all edges
  edge2triangles = defaultdict(list) # map of edge -> list of triangles
  for tri in triangles:
    edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
    for edge_i, edge_j in edges:
      i, j = min(edge_i, edge_j), max(edge_i, edge_j)
      edge2triangles[(i,j)].append(tri)
      if len(edge2triangles[(i,j)]) > 1: continue
      sys.add_distance_constraint(i, j, np.linalg.norm(init_pos[i] - init_pos[j]))
  edges = np.array(edge2triangles.keys())
  print 'number of edges', len(edges)

  # add bending constraints between all pairs of adjacent triangles
  num_bending_constraints = 0
  for triangles in edge2triangles.itervalues():
    if len(triangles) == 1: continue # this edge lies on the cloth border
    assert len(triangles) == 2 # non-border edges must be a part of two triangles
    tri1, tri2 = triangles
    common_nodes = np.intersect1d(tri1, tri2)
    assert len(common_nodes) == 2
    p1, p2, p3, p4 = common_nodes[0], common_nodes[1], np.setdiff1d(tri1, common_nodes)[0], np.setdiff1d(tri2, common_nodes)[0]
    n1 = np.cross(init_pos[p2] - init_pos[p1], init_pos[p3] - init_pos[p1]); n1 /= np.linalg.norm(n1)
    n2 = np.cross(init_pos[p2] - init_pos[p1], init_pos[p4] - init_pos[p1]); n2 /= np.linalg.norm(n2)
    sys.add_bending_constraint(p1, p2, p3, p4, np.arccos(n1.dot(n2)))
    num_bending_constraints += 1
  print 'number of bending constraints', num_bending_constraints

  sys.randomize_constraints()
  return sys, edges


class TriangleMesh(object):
  def __init__(self, sys, edges):
    self.sys, self.edges = sys, edges

  @classmethod
  def FromObjFile(cls, obj_file, init_center, total_mass, sim_params):
    vertices, triangles = [], []
    with open(obj_file, 'r') as f:
      for _line in f:
        line = _line.strip()
        if not line: continue
        if line[0] == '#': continue
        if line[0] == 'v':
          parts = line.split(); assert len(parts) == 4
          _, x, y, z = parts
          vertices.append([float(x), float(y), float(z)])
        elif line[0] == 'f':
          parts = line.split(); assert len(parts) == 4
          _, i, j, k = parts
          triangles.append([int(i)-1, int(j)-1, int(k)-1])

    init_pos = np.array(vertices)
    init_pos -= init_pos.mean(axis=0) - init_center
    triangles = np.array(triangles, dtype='int32')
    num_nodes = len(init_pos)
    masses = np.ones(num_nodes) * (float(total_mass) / num_nodes)
    sys, edges = make_mass_system(init_pos, masses, triangles, sim_params)
    out = cls(sys, edges)
    assert np.allclose(sys.get_triangles(), triangles)
    return out

  def get_num_nodes(self): return self.sys.get_num_nodes()
  def step(self): self.sys.step()
  def get_sys(self): return self.sys
  def get_node_positions(self): return self.sys.get_node_positions()
  def get_edge_positions(self): return self.get_node_positions()[self.edges]
  def dump_obj_data(self):
    out = []
    for v in self.get_node_positions():
      out.append('v %f %f %f' % (v[0], v[1], v[2]))
    for tri in self.sys.get_triangles():
      out.append('f %d %d %d' % (tri[0]+1, tri[1]+1, tri[2]+1))
    return '\n'.join(out)

class Cloth(TriangleMesh):
  def __init__(self, res_x, res_y, len_x, len_y, init_center, total_mass, sim_params):
    self.res_x, self.res_y, self.len_x, self.len_y, self.init_center = res_x, res_y, len_x, len_y, init_center

    # initialize node positions
    assert self.res_x >= 2 and self.res_y >= 2
    self.num_nodes = self.res_x * self.res_y

    init_pos = np.zeros((self.num_nodes, 3))
    init_pos[:,:2] = np.dstack(np.meshgrid(np.linspace(0, self.len_x, self.res_x), np.linspace(0, self.len_y, self.res_y))).reshape((-1, 2))
    init_pos -= init_pos.mean(axis=0) - self.init_center
    masses = np.ones(self.num_nodes) * (float(total_mass) / self.num_nodes)

    # declare which triples of nodes constitute triangles
    # (currently only used for raycasting)
    num_triangles = 2 * (self.res_x-1) * (self.res_y-1)
    triangles = np.empty((num_triangles, 3), dtype='int32')
    curr = 0
    for i in range(self.num_nodes):
      x, y = self._i_to_xy(i)
      if x+1 < self.res_x and y+1 < self.res_y:
        # switch up which way to draw the triangle hypotenuse
        # to break symmetries in the bending constraints
        if (curr / 2) % 2 == 0:
          triangles[curr,:]  = [i, self._xy_to_i(x+1,y), self._xy_to_i(x,y+1)]
          triangles[curr+1,:] = [self._xy_to_i(x+1,y), self._xy_to_i(x+1,y+1), self._xy_to_i(x,y+1)]
        else:
          triangles[curr,:] = [i, self._xy_to_i(x+1,y+1), self._xy_to_i(x,y+1)]
          triangles[curr+1,:] = [i, self._xy_to_i(x+1,y), self._xy_to_i(x+1,y+1)]
        curr += 2
    assert curr == num_triangles

    sys, edges = make_mass_system(init_pos, masses, triangles, sim_params)
    TriangleMesh.__init__(self, sys, edges)

  def _i_to_xy(self, i): return i % self.res_x, i // self.res_x
  def _xy_to_i(self, x, y): return y*self.res_x + x
