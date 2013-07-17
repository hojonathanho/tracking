import numpy as np
import trackingpy

# initialize node positions
M, N = 10, 15
init_width = .5
init_height = 1
init_center = np.array([0, 0, 0])
init_z = 0

init_pos = np.zeros((N*M, 3))
init_pos[:,:2] = np.dstack(np.meshgrid(np.linspace(0, init_width, M), np.linspace(0, init_height, N))).reshape((-1, 2))
init_pos[:,2] = init_z
init_pos -= init_pos.mean(axis=0) - init_center
masses = np.ones(M*N)

p = trackingpy.SimulationParams()
p.dt = .01
p.solver_iters = 10
print '========================================='
p.gravity = np.array([0, 0, -9.8])
print 'py gravity', p.gravity
print '========================================='

cloth = trackingpy.Cloth(init_pos, masses, p)
for i in range(M):
  cloth.add_anchor_constraint(i, init_pos[i])

def i_to_xy(i):
    return i % M, i / M
def xy_to_i(x, y):
    return y*M + x
coords = [(i,) + i_to_xy(i) for i in range(N*M)]

constraints_to_cut = []

for i, x, y in coords:
  cnts = []
  if x-1 >= 0:
    j = xy_to_i(x-1,y)
    cnts.append(cloth.add_distance_constraint(i, j, np.linalg.norm(init_pos[i] - init_pos[j])))
  if x+1 < M:
    j = xy_to_i(x+1,y)
    cnts.append(cloth.add_distance_constraint(i, j, np.linalg.norm(init_pos[i] - init_pos[j])))
  if y+1 < N:
    j = xy_to_i(x,y+1)
    cnts.append(cloth.add_distance_constraint(i, j, np.linalg.norm(init_pos[i] - init_pos[j])))
  if y-1 >= 0:
    j = xy_to_i(x,y-1)
    cnts.append(cloth.add_distance_constraint(i, j, np.linalg.norm(init_pos[i] - init_pos[j])))
  if x+1 < M and y+1 < N:
    j = xy_to_i(x+1,y+1)
    cnts.append(cloth.add_distance_constraint(i, j, np.linalg.norm(init_pos[i] - init_pos[j])))
  if x-1 >= 0 and y-1 >= 0:
    j = xy_to_i(x-1,y-1)
    cnts.append(cloth.add_distance_constraint(i, j, np.linalg.norm(init_pos[i] - init_pos[j])))
  if x+1 < M and y-1 >= 0:
    j = xy_to_i(x+1,y-1)
    cnts.append(cloth.add_distance_constraint(i, j, np.linalg.norm(init_pos[i] - init_pos[j])))
  if x-1 >= 0 and y+1 < N:
    j = xy_to_i(x-1,y+1)
    cnts.append(cloth.add_distance_constraint(i, j, np.linalg.norm(init_pos[i] - init_pos[j])))

  if y == N/2 or y+1 == N/2 or y-1 == N/2:
    constraints_to_cut += cnts

def cut():
  for i in constraints_to_cut:
    cloth.disable_constraint(i)

import openravepy
import trajoptpy
env = openravepy.Environment()
viewer = trajoptpy.GetViewer(env)


# from timeit import Timer
# print 'timing'
# t = Timer(lambda: cloth.step())
# print t.timeit(number=10000)
# raw_input('done timing')

heights = []
for i in range(100000):
  print i
  cloth.step()
  h = env.plot3(cloth.get_node_positions(), 5)
  viewer.Step()
  heights.append(cloth.get_node_positions()[0,2])

  if i == 100:
    cut()

print heights

import matplotlib.pyplot as plt
plt.plot(range(len(heights)), heights)
plt.show()
