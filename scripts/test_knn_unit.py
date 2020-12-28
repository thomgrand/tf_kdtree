"""Script that tests the compiled TF-KDTree
"""
import sys
sys.path.append("../") #TODO: Hack

import os
import unittest
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
assert(not tf.executing_eagerly())
from tf_nearest_neighbor import nn_distance, buildKDTree, searchKDTree
import sys
from IPython.utils import io
import timeit
from scipy.spatial import cKDTree

np.random.seed(0)

class TestKNNImplementation(unittest.TestCase):

  def __init__(self, test_name = None):
    super(TestKNNImplementation, self).__init__(test_name)
    self.sess = tf.compat.v1.InteractiveSession()


  def referenceSolution(self, points_ref, points_query, k):
    kdtree = cKDTree(points_ref)
    dists, inds = kdtree.query(points_query, k)

    return dists, inds

  def executeTest(self, nr_refs, nr_query, k, d=3):

    points_ref = np.random.uniform(size=(nr_refs, d)).astype(np.float32) * 1e3
    points_query = np.random.uniform(size=(nr_query, d)).astype(np.float32) * 1e3

    points_ref_tf = tf.compat.v1.placeholder(dtype=tf.float32, shape=points_ref.shape)
    points_query_tf = tf.compat.v1.placeholder(dtype=tf.float32, shape=points_query.shape)

    dists_ref, inds_ref = self.referenceSolution(points_ref, points_query, k=k)
    nn_distance_result = nn_distance(points_ref, points_query, nr_nns_searches=k)

    dists_knn, inds_knn = self.sess.run(nn_distance_result, feed_dict={points_ref_tf: points_ref, points_query_tf: points_query})

    #Shape checks
    self.assertTrue(inds_knn.shape[-1] == k)
    self.assertTrue(inds_knn.shape[0] == points_query.shape[0])
    self.assertTrue(np.all(inds_knn.shape == dists_knn.shape))
    self.assertTrue((dists_ref.ndim == 1 and dists_knn.ndim == 2 and dists_knn.shape[-1] == 1)
                    or np.all(dists_ref.shape == dists_knn.shape))

    self.checkSuccessful(points_ref, points_query, k, dists_ref, inds_ref, dists_knn, inds_knn)

  def checkSuccessful(self, points_ref, points_query, k, dists_ref, inds_ref, dists_knn, inds_knn):

    if dists_ref.ndim == 1:
      #dists_knn = dists_knn[..., 0]
      #inds_knn = inds_knn[..., 0]
      dists_ref = dists_ref[..., np.newaxis]
      inds_ref = inds_ref[..., np.newaxis]

    self.assertTrue(
      np.allclose(dists_ref ** 2, np.sum((points_query[:, np.newaxis] - points_ref[inds_ref]) ** 2, axis=-1),
                  atol=1e-5))
    self.assertTrue(
      np.allclose(dists_knn, np.sum((points_query[:, np.newaxis] - points_ref[inds_knn]) ** 2, axis=-1), atol=1e-5))
    self.assertTrue(
      np.allclose(dists_ref ** 2, np.sum((points_query[:, np.newaxis] - points_ref[inds_knn]) ** 2, axis=-1),
                  atol=1e-5))
    self.assertTrue(
      np.allclose(dists_knn, np.sum((points_query[:, np.newaxis] - points_ref[inds_ref]) ** 2, axis=-1), atol=1e-5))

    self.assertTrue(np.allclose(dists_ref ** 2, dists_knn, atol=1e-5), "Mismatch in KNN-Distances")

    # For larger values this sometimes flip
    # if k <= 100 and nr_query < 1e5 and nr_refs < 1e5:
    #  self.assertTrue(np.all(inds_ref == inds_knn), "Mismatch in KNN-Indices")
    # else:
    self.assertTrue(np.sum(inds_ref == inds_knn) / inds_ref.size > 0.999, "Too many mismatches in KNN-Indices")

  def test_small_equal_size(self):
    for i in range(10, 31):
      self.executeTest(i, i, np.minimum(i, 10))

  def test_small_size(self):
    for i in range(1, 21):
      self.executeTest(10, i, 1)

    for i in range(10, 31):
      self.executeTest(int(i), 5, np.minimum(i, 10))

  #"""
  def test_large_size(self):
    for i in np.logspace(3, 6, 10):
      self.executeTest(10, int(i), 1)

    for i in np.logspace(3, 6, 10):
      self.executeTest(int(i), 5, np.minimum(int(i), 10))

  #def test_large_knn_size(self):
  #  for i in np.logspace(1.5, 3, 10):
  #    self.executeTest(10000, 1000, int(i))
  #"""

  def test_stress_test(self):
    self.executeTest(int(1e5), int(1e5), 50)


class TestKDTreeImplementation(TestKNNImplementation):

  def __init__(self, test_name = None):
    super(TestKDTreeImplementation, self).__init__(test_name)

  def executeTest(self, nr_refs, nr_query, k, d=3):

    points_ref = np.random.uniform(size=(nr_refs, d)).astype(np.float32) * 1e3
    points_query = np.random.uniform(size=(nr_query, d)).astype(np.float32) * 1e3

    #import pdb; pdb.set_trace()
    points_ref_tf = tf.compat.v1.placeholder(dtype=tf.float32, shape=points_ref.shape)
    points_query_tf = tf.compat.v1.placeholder(dtype=tf.float32, shape=points_query.shape)

    build_kdtree_op = buildKDTree(points_ref_tf, levels=None) #(3 if nr_refs < 1500 else 8))

    structured_points, part_nr, shuffled_inds = self.sess.run(build_kdtree_op, feed_dict={points_ref_tf: points_ref})
    kdtree_results = searchKDTree(points_query_tf, part_nr[0], nr_nns_searches=k, shuffled_inds=shuffled_inds.astype(np.int32))

    dists_ref, inds_ref = self.referenceSolution(points_ref, points_query, k=k)
    dists_knn, inds_knn = self.sess.run(kdtree_results, feed_dict={points_query_tf: points_query})
    #inds_knn = shuffled_inds[inds_knn]

    #Shape checks
    self.assertTrue(inds_knn.shape[-1] == k)
    self.assertTrue(inds_knn.shape[0] == points_query.shape[0])
    self.assertTrue(np.all(inds_knn.shape == dists_knn.shape))
    self.assertTrue((dists_ref.ndim == 1 and dists_knn.ndim == 2 and dists_knn.shape[-1] == 1)
                    or np.all(dists_ref.shape == dists_knn.shape))

    self.checkSuccessful(points_ref, points_query, k, dists_ref, inds_ref, dists_knn, inds_knn)

    #import pdb; pdb.set_trace()


if __name__ == "__main__":
  with tf.device("/gpu:0"):
    unittest.main()

  #with tf.device("/gpu:0"):
  #  unittest.main()
