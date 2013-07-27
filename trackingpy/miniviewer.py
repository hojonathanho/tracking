import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import threading
import Queue

def normalized(v):
  return v / np.linalg.norm(v)

def rotation_mat(axis, angle):
  assert len(axis) == 3
  axis = normalized(axis)
  crossprod_axis = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
  s, c = np.sin(angle), np.cos(angle)
  m = c*np.eye(3) + s*crossprod_axis + (1.-c)*np.outer(axis, axis)
  return m

class Handle(object):
  def __init__(self):
    self.dirty = True
    self.viewer = None
    self.lst = None
    self.marked_for_delete = False
    self.deleted = False

  def init_for_viewer(self, viewer):
    self.viewer = viewer
    self.lst = glGenLists(1)
    self.dirty = True

  def draw(self):
    assert self.viewer is not None
    if self.dirty:
      self.force_update()
    else:
      glCallList(self.lst)

  def force_update(self):
    glNewList(self.lst, GL_COMPILE_AND_EXECUTE)
    self.on_update()
    glEndList()
    self.dirty = False

  def delete_if_requested(self, force=False):
    if self.lst is not None and (self.marked_for_delete or force) and not self.deleted:
      glDeleteLists(self.lst, 1)
      self.deleted = True

  def on_update(self):
    pass


class ArrowHandle(Handle):
  def __init__(self, start, end, radius, color):
    assert len(start) == 3 and len(end) == 3
    if color is not None: assert len(color) in [3, 4]
    super(ArrowHandle, self).__init__()
    self.start, self.end, self.radius, self.color = start, end, radius, color
    self.arrow_dir = normalized(self.end - self.start)

  def on_update(self):
    if self.color is not None: glColor(*self.color)

    rot_angle = np.degrees(np.arccos(np.array([0, 0, 1]).dot(self.arrow_dir)))
    rot_axis = normalized(np.cross(np.array([0, 0, 1]), self.arrow_dir))

    # cone tip
    cone_height = 2*self.radius
    self.cone_quadric = gluNewQuadric()
    gluQuadricNormals(self.cone_quadric, GLU_SMOOTH)
    glPushMatrix()
    glTranslate(*(self.end - cone_height*self.arrow_dir))
    glRotate(rot_angle, rot_axis[0], rot_axis[1], rot_axis[2])
    gluCylinder(self.cone_quadric, cone_height, 0, cone_height, 32, 32)
    glPopMatrix()

    # cylinder body
    cyl_height = max(0, np.linalg.norm(self.end - self.start) - cone_height)
    self.cyl_quadric = gluNewQuadric()
    gluQuadricNormals(self.cyl_quadric, GLU_SMOOTH)
    glPushMatrix()
    glTranslate(*self.start)
    glRotate(rot_angle, rot_axis[0], rot_axis[1], rot_axis[2])
    gluCylinder(self.cyl_quadric, self.radius, self.radius, cyl_height, 32, 32)
    glPopMatrix()

    # sphere bottom
    # self.sphere_quadric = gluNewQuadric()
    # gluQuadricNormals(self.sphere_quadric, GLU_SMOOTH)
    # glPushMatrix()
    # glTranslate(*self.start)
    # gluSphere(self.sphere_quadric, 1.5*self.radius, 32, 32)
    # glPopMatrix()

class AxesHandle(Handle):
  def __init__(self, T, size):
    super(AxesHandle, self).__init__()
    assert T.shape == (4, 4)
    self.T, self.size = T, size
    self.arrows = []
    p0 = T[:3,3]
    xax, yax, zax = T[:3,:3].T*size
    width = size/10.
    self.arrows.append(ArrowHandle(p0, p0+xax, width, [1,0,0]))
    self.arrows.append(ArrowHandle(p0, p0+yax, width, [0,1,0]))
    self.arrows.append(ArrowHandle(p0, p0+zax, width, [0,0,1]))

  def on_update(self):
    for a in self.arrows:
      a.on_update()

class BoxHandle(Handle):
  def __init__(self, center, extents, color):
    assert len(center) == 3 and len(extents) == 3
    if color is not None: assert len(color) in [3, 4]
    super(BoxHandle, self).__init__()
    self.center, self.extents, self.color = center, extents, color

  def on_update(self):
    if self.color is not None: glColor(*self.color)
    glPushMatrix()
    glTranslate(*(self.center))
    glScale(*self.extents*2)
    glTranslate(-.5, -.5, .5)
    glBegin(GL_QUADS)
    # front
    glVertex3f(0., 0., 0.)
    glVertex3f(1., 0., 0.)
    glVertex3f(1., 1., 0.)
    glVertex3f(0., 1., 0.)
    # back
    glVertex3f(0., 0., -1.)
    glVertex3f(1., 0., -1.)
    glVertex3f(1., 1., -1.)
    glVertex3f(0., 1., -1.)
    # right
    glVertex3f(1., 0., 0.)
    glVertex3f(1., 0., -1.)
    glVertex3f(1., 1., -1.)
    glVertex3f(1., 1., 0.)
    # left
    glVertex3f(0., 0., 0.)
    glVertex3f(0., 0., -1.)
    glVertex3f(0., 1., -1.)
    glVertex3f(0., 1., 0.)
    # top
    glVertex3f(0., 1., 0.)
    glVertex3f(1., 1., 0.)
    glVertex3f(1., 1., -1.)
    glVertex3f(0., 1., -1.)
    # bottom
    glVertex3f(0., 0., 0.)
    glVertex3f(1., 0., 0.)
    glVertex3f(1., 0., -1.)
    glVertex3f(0., 0., -1.)
    glEnd()
    glPopMatrix()


class GLPrimitiveHandle(Handle):
  def __init__(self, gl_mode, vertices, indices, normals, colors):
    assert indices.ndim == 1
    assert vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) == len(indices)
    if normals is not None: assert normals.ndim == 2 and normals.shape[1] == 3 and len(normals) == len(indices)
    if colors is not None:
      colors = np.asarray(colors)
      if colors.ndim < 2:
        colors = np.tile(colors, len(indices)).reshape((-1, len(colors)))
      assert colors.ndim == 2 and colors.shape[1] in [3, 4] and len(colors) == len(indices)

    super(GLPrimitiveHandle, self).__init__()
    self.gl_mode = gl_mode
    self.vertices = vertices
    self.normals = normals
    self.colors = colors
    self.indices = indices

  def on_update(self):
      glEnableClientState(GL_VERTEX_ARRAY)
      glVertexPointerf(self.vertices)

      if self.colors is not None:
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointerf(self.colors)

      if self.normals is not None:
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointerf(self.normals)

      glDrawElementsui(self.gl_mode, self.indices)

      glDisableClientState(GL_VERTEX_ARRAY)
      if self.colors is not None: glDisableClientState(GL_COLOR_ARRAY)
      if self.normals is not None: glDisableClientState(GL_NORMAL_ARRAY)

class LinesHandle(GLPrimitiveHandle):
  def __init__(self, lines, size, colors):
    self.size = size
    super(LinesHandle, self).__init__(GL_LINES, lines, np.arange(len(lines)), None, colors)

  def on_update(self):
    glLineWidth(self.size)
    super(LinesHandle, self).on_update()

class PointsHandle(GLPrimitiveHandle):
  def __init__(self, points, size, colors):
    self.size = size
    super(PointsHandle, self).__init__(GL_POINTS, points, np.arange(len(points)), None, colors)

  def on_update(self):
    glPointSize(self.size)
    super(PointsHandle, self).on_update()

class HandleWrapper(object):
  def __init__(self, handle):
    assert isinstance(handle, Handle)
    self.handle = handle

  def __del__(self):
    self.handle.marked_for_delete = True

class Viewer(object):
  def __init__(self, thread_host, w=640, h=480, fov=45):
    self.w, self.h, self.fov = w, h, fov

    # viewer state
    self.view_prev_sx, self.view_prev_sy = 0, 0
    self.view_prev_mouse_pressed = False

    self.running = False

    # handles
    self.thread_host = thread_host
    self.handles = []
    self.handles_to_add = Queue.Queue()
    self.handles_being_added = False

    self._assert_host_thread()
    pygame.init()
    self._set_display_mode()

  def _set_display_mode(self):
    pygame.display.set_mode((self.w, self.h), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)

  def _assert_host_thread(self):
    assert self.thread_host is not None and threading.current_thread().name == HostThread.THREAD_NAME

  def _assert_not_host_thread(self):
    assert self.thread_host is not None and threading.current_thread().name != HostThread.THREAD_NAME

  def init_scene(self):
    # glEnable(GL_LIGHTING)
    # glEnable(GL_COLOR_MATERIAL)
    # glShadeModel(GL_SMOOTH)
    # glEnable(GL_LIGHT0)
    # glLightfv(GL_LIGHT0, GL_DIFFUSE, (1., 1., 1., 1.0))
    # glLightfv(GL_LIGHT0, GL_POSITION, (0.0, 0.0, 0.0, 10.0))

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    self.cam_pos = np.array([1., 1., 0.])
    self.cam_target = np.array([0., 0., 0.])
    self.cam_up = np.array([0., 0., 1.])

    self.bg_color = np.array([1, 1, 1])

  # View/camera methods (internal)
  def _set_view(self):
    # orthogonalize
    cam_dir = normalized(self.cam_target - self.cam_pos)
    self.cam_up = normalized(np.cross(np.cross(cam_dir, self.cam_up), cam_dir))
    gluLookAt(
      self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
      self.cam_target[0], self.cam_target[1], self.cam_target[2],
      self.cam_up[0], self.cam_up[1], self.cam_up[2]
    )

  def _cam_to_world_mat(self):
    m = np.empty((3,3))
    up = m[:,1] = normalized(self.cam_up)
    cam_dir = m[:,2] = normalized(self.cam_target - self.cam_pos)
    m[:,0] = np.cross(cam_dir, up)
    return m

  def _to_arcball_vec(self, sx, sy):
    v = np.array([2.*(sx + .5)/self.w - 1., -2.*(sy + .5)/self.h + 1., 0])
    norm2 = (v[:2]**2).sum()
    if norm2 <= 1: v[2] = np.sqrt(1. - norm2)
    else: v /= np.sqrt(norm2)
    return v

  def _process_input(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
        self.running = False

      elif event.type == pygame.VIDEORESIZE:
        self.w, self.h = event.w, event.h
        self._set_display_mode()

      elif event.type == pygame.MOUSEMOTION:
        if any(pygame.mouse.get_pressed()):
          sx, sy = pygame.mouse.get_pos()
          v_old = self._to_arcball_vec(self.view_prev_sx, self.view_prev_sy)
          if self.view_prev_mouse_pressed:
            v_new = self._to_arcball_vec(sx, sy)
            angle = np.arccos(v_old.dot(v_new))
            if angle >= .0001:
              # rotation around camera target
              if pygame.mouse.get_pressed()[0]:
                m = self._cam_to_world_mat()
                rot = rotation_mat(np.cross(v_old, v_new), angle)
                self.cam_pos = m.dot(rot).dot(m.T).dot(self.cam_pos)
              # panning (shift both camera position and target by same amount)
              elif pygame.mouse.get_pressed()[1]:
                curr_dist = np.linalg.norm(self.cam_target - self.cam_pos)
                sdiff = np.array([-sx + self.view_prev_sx, sy - self.view_prev_sy, 0.]) / float(max(self.w, self.h)) * curr_dist
                pan_diff = self._cam_to_world_mat().dot(sdiff)
                self.cam_pos += pan_diff
                self.cam_target += pan_diff
              # zooming (translation to/from camera target)
              elif pygame.mouse.get_pressed()[2]:
                curr_dist = np.linalg.norm(self.cam_target - self.cam_pos)
                diff = curr_dist*angle # scale movement speed by current distance
                new_dist = curr_dist + diff*(-1. if sy > self.view_prev_sy else 1.)
                if new_dist > .01:
                  self.cam_pos = normalized(self.cam_pos - self.cam_target)*new_dist + self.cam_target
              self._set_view()
          self.view_prev_sx, self.view_prev_sy = sx, sy
          self.view_prev_mouse_pressed = True
        else:
          self.view_prev_mouse_pressed = False

  def draw(self):
    # add the handles in the handles-to-add queue
    if not self.handles_to_add.empty():
      self.handles_being_added = True
      try:
        while True:
          h = self.handles_to_add.get_nowait()
          assert isinstance(h, Handle)
          h.init_for_viewer(self)
          self.handles.append(h)
      except Queue.Empty:
        pass
      self.handles_being_added = False

    # get rid of handles marked deleted
    for h in self.handles:
      h.delete_if_requested()
    self.handles = [h for h in self.handles if not h.deleted]

    # draw
    glClearColor(self.bg_color[0], self.bg_color[1], self.bg_color[2], 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glViewport(0, 0, self.w, self.h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(self.fov, float(self.w)/self.h, .0001, 10000.)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    self._set_view()

    for h in self.handles:
      h.draw()

  def step(self):
    pygame.display.flip()
    self._process_input()
    self.draw()

  def loop(self):
    self.running = True
    while self.running:
      self.step()

  def wait(self):
    self._assert_not_host_thread()
    while self.running:
      time.sleep(.01)

  def add_handle(self, hw):
    self._assert_not_host_thread()
    assert isinstance(hw, HandleWrapper)
    self.handles_to_add.put(hw.handle)
    return hw

  def block_until_all_drawn(self):
    while self.handles_being_added or any(h.dirty for h in self.handles):
      time.sleep(.001)

  # Handle construction convenience functions
  def plot3(self, points, size, colors=None):
    return self.add_handle(HandleWrapper(PointsHandle(points, size, colors)))

  def drawlinelist(self, lines, size, colors=None):
    return self.add_handle(HandleWrapper(LinesHandle(lines, size, colors)))

  def drawarrow(self, start, end, radius, color=None):
    return self.add_handle(HandleWrapper(ArrowHandle(start, end, radius, color)))

  def drawbox(self, center, extents, color=None):
    return self.add_handle(HandleWrapper(BoxHandle(center, extents, color)))

  def drawaxes(self, T, size):
    return self.add_handle(HandleWrapper(AxesHandle(T, size)))

class HostThread(threading.Thread):
  THREAD_NAME = '__viewer_host_thread__'

  def __init__(self):
    super(HostThread, self).__init__(name=self.THREAD_NAME)
    self.viewer = None

  def run(self):
    self.viewer = Viewer(self)
    self.viewer.init_scene()
    self.viewer.loop()

def make_viewer():
  t = HostThread()
  t.setDaemon(True)
  t.start()
  while t.viewer is None or not t.viewer.running:
    time.sleep(.01)
  return t.viewer


def main():
  viewer = make_viewer()

  import cloudprocpy
  from trackingpy import clouds
  grabber = cloudprocpy.CloudGrabber()
  grabber.startRGBD()
  while True:
    rgb, depth = grabber.getRGBD()
    xyz = clouds.depth_to_xyz(depth, 1/1000.).reshape((-1, 3))
    rgb = rgb.astype(float).reshape((-1, 3)); rgb /= 255.
    h = viewer.plot3(xyz, 1, rgb)
    viewer.block_until_all_drawn()

if __name__ == '__main__':
  main()
