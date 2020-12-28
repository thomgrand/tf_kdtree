import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
_op_library = tf.load_op_library(
    os.path.join(os.path.dirname(__file__), 'libtf_nndistance.so'))

def nn_distance(points_ref, points_query, nr_nns_searches=1):
    """Simple KNN implementation by comparing all points.

    This function implements a simple KNN computation by comparing all
    points in the two point clouds points_ref and points_query. Note that
    this operation is quadratic in time, but linear in memory.

    Parameters
    ----------
    points_ref : tensor or array of float or double precision
        Points in which the KNNs will be searched
    points_query : tensor or array of float or double precision
        Points which will search for their KNNs
    nr_nns_searches : int, optional
        How many closest nearest neighbors will be queried (=k), by default 1

    Returns
    -------
    tuple
        Returns the tuple containing

        * dists (tensor of float or double precision) : Quadratic distance of KD-Tree points to the queried points
        * inds (tensor of type int) : Indices of the K closest neighbors
    """
    if points_ref.shape[-1] == 2:
      assert(points_query.shape[-1] == 2)
      points_ref = tf.cast(points_ref, tf.float32)
      points_query = tf.cast(points_query, tf.float32)
      points_ref = tf.concat([points_ref, tf.zeros_like(points_ref[..., -1:])], axis=-1)
      points_query = tf.concat([points_query, tf.zeros_like(points_query[..., -1:])], axis=-1)

    return _op_library.knn_distance(points_ref, points_query, nr_nns_searches=nr_nns_searches)


def buildKDTree(points_ref, levels=None, **kwargs):
    """Builds the KD-Tree for subsequent queries using searchKDTree

    Builds the KD-Tree for subsequent queries using searchKDTree. Note that the 
    tree is always built on the CPU and then transferred to the GPU if necessary.

    Parameters
    ----------
    points_ref : tensor or array of float or double precision
        Points from which to build the KD-Tree
    levels : int, optional
        Levels of the KD-Tree (currently between 1 and 13 levels). If none is specified, will pick an appropriate value.

    Returns
    -------
    tuple
        Returns a triplet 
        
        * structured_points: points_ref, ordered by the KD-Tree structure
        * part_nr: Unique ID of the KD-Tree to later refer to the created tree
        * shuffled_ind: Indices to map structured_points -> points_ref
    """
    if levels is None:
      levels = np.maximum(1, np.minimum(13, int(np.log(int(points_ref.shape[0])) / np.log(2))-3))

    with tf.device("/cpu:0"):
      return _op_library.build_kd_tree(points_ref, levels=levels, **kwargs)


def searchKDTree(points_query, metadata_address_kdtree, nr_nns_searches=1, shuffled_inds=None, **kwargs):
    """Searches the specified KD-Tree for KNN of the given points

    Parameters
    ----------
    points_query : tensor or array of float or double precision
        Points for which the KNNs will be computed
    metadata_address_kdtree : int
        Unique ID of the KD-Tree to be queried (see buildKDTree)
    nr_nns_searches : int, optional
        How many closest nearest neighbors will be queried (=k), by default 1
    shuffled_inds : tensor or array of type int, optional
        When creating the tree using buildKDTree, this array is returned to map
        the indices from structured_points, back to the original indices.
        If none, this remapping will not be performed and the returned indices
        map to the indices in structured_points.

    Returns
    -------
    tuple
        Returns the tuple containing

        * dists (tensor of float or double precision) : Quadratic distance of KD-Tree points to the queried points
        * inds (tensor of type int) : Indices of the K closest neighbors
    """
    dists, inds = _op_library.kd_tree_knn_search(points_query, metadata_address_kdtree=metadata_address_kdtree, 
                nr_nns_searches=nr_nns_searches, **kwargs)

    if shuffled_inds is not None:
        inds = tf.gather(shuffled_inds, tf.cast(inds, tf.int32))

    return dists, inds
