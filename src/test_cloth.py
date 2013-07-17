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
p.solver_iters = 2
print '========================================='
p.gravity = np.array([0, 0, -9.8])
print 'py gravity', p.gravity
print '========================================='

cloth = trackingpy.Cloth(init_pos, masses, p)


import openravepy
import trajoptpy
env = openravepy.Environment()
viewer = trajoptpy.GetViewer(env)

heights = []
for i in range(100):
  cloth.step()
  h = env.plot3(cloth.getNodePositions(), 5)
  viewer.Step()
  heights.append(cloth.getNodePositions()[0,2])

print heights

import matplotlib.pyplot as plt
plt.plot(range(len(heights)), heights)
plt.show()
