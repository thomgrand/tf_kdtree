"""This script computes the data for the benchmark
"""
import numpy as np

ks = np.array([1, 10, 100])
nr_refs = np.logspace(2, 6, num=15).astype(np.int32)
nr_queries = np.logspace(1, 6, num=15).astype(np.int32)

if __name__ == "__main__":
    from IPython import get_ipython
    from scipy.spatial import cKDTree
    import sys
    sys.path.append("../") #TODO: Hack
    from tf_nearest_neighbor import nn_distance, buildKDTree, searchKDTree
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True) #Only allocate as much memory as is needed
        except RuntimeError as e:
            print(e)
    import numpy as np

    #tf.compat.v1.disable_eager_execution()
    ipython = get_ipython()
    #sess = tf.compat.v1.InteractiveSession()

    #ret = ipython.run_line_magic("timeit", "-o -q abs(-42)")
    #ret.average, ret.stdev

    def simple_nn(xyz1, xyz2, k):
        diff = xyz1[np.newaxis] - xyz2[:, np.newaxis]
        square_dst = np.sum(diff**2, axis=-1)
        dst = np.sort(square_dst, axis=1)[..., :k]
        idx = np.argsort(square_dst, axis=1)[..., :k]
        return dst, idx

    dims = 3

    timing_results = np.zeros(shape=[3, nr_refs.size, nr_queries.size, ks.size])

    for ref_i, nr_ref in enumerate(nr_refs):
        #Create data
        points_ref = np.random.uniform(size=(nr_ref, dims)).astype(np.float32) * 1e3
        points_ref_tf = tf.convert_to_tensor(points_ref)

        #Build KD-Trees right here to save some time
        kdtree = cKDTree(points_ref)
        structured_points, part_nr, shuffled_inds = buildKDTree(points_ref_tf, levels=None)

        for query_i, nr_query in enumerate(nr_queries):
            points_query = np.random.uniform(size=(nr_query, dims)).astype(np.float32) * 1e3
            points_query_tf = tf.convert_to_tensor(points_query)

            for k_i, k in enumerate(ks):        
                print("------- {}, {}, {} --------".format(ref_i, query_i, k_i))
                #Simple numpy implementation
                #Skip if too large (>= 4GB)
                #if points_ref.nbytes * points_query.nbytes >= 4e9:
                #    timing_results[0, ref_i, query_i, k_i] = np.nan
                #else:
                    #Use -q to supress the output
                #    timing = ipython.run_line_magic("timeit", "-o simple_nn(points_ref, points_query, k)")
                #    timing_results[0, ref_i, query_i, k_i] = timing.average
                timing = ipython.run_line_magic("timeit", "-o nn_distance(points_ref_tf, points_query_tf, nr_nns_searches=k)")
                timing_results[0, ref_i, query_i, k_i] = timing.average
                    
                #Scipy spatial implementation                
                timing = ipython.run_line_magic("timeit", "-o kdtree.query(points_query, k)")
                timing_results[1, ref_i, query_i, k_i] = timing.average

                #TF KD-Tree implementation 
                searchKDTree(points_query, part_nr[0], nr_nns_searches=k, shuffled_inds=shuffled_inds)
                #Build the graph first to avoid measuring this overhead
                #sess.run(kdtree_results, feed_dict={points_query_tf: points_query})
                #timing = ipython.run_line_magic("timeit", "-o sess.run(kdtree_results, feed_dict={points_query_tf: points_query})")
                timing = ipython.run_line_magic("timeit", "-o searchKDTree(points_query, part_nr[0], nr_nns_searches=k, shuffled_inds=shuffled_inds)")
                timing_results[2, ref_i, query_i, k_i] = timing.average


    np.savez_compressed("benchmark_results.npz", timing_results=timing_results)

