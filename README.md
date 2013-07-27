Tracking Deformable Objects with Point Clouds
========

This is an implementation of the tracking algorithm of the paper

    Tracking Deformable Objects with Point Clouds
    John D. Schulman, Alex Lee, Jonathan Ho and Pieter Abbeel
    In the proceedings of the International Conference on Robotics and Automation (ICRA), 2013.

(website: http://rll.berkeley.edu/tracking/)

This implementation tracks cloth only, using a position-based dynamics model (Mueller et al. 2007). The cloth simulation does not support self-collisions, unlike the Bullet-based simulation that was used in the original implementation described in the paper.

Direct dependencies:

  - Python 2.7
  - PyOpenGL >= 3.0.1
  - pygame >= 1.9.1
  - python-opencv (cv2)
  - h5py
  - numpy
  - PCL >= 1.6
  - Boost >= 1.48, including Boost Python
  - Eigen 3

  For GPU/CUDA support:
  - PyCUDA >= 2012.1

  For live tracking for a depth sensor on the head of a PR2,
  - rapprentice (http://rll.berkeley.edu/rapprentice) and its dependencies
