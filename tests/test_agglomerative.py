import dataclasses
import unittest

import numpy as np

from cltl.vector_id.clusterid import ClusterIdentity


@dataclasses.dataclass
class TestCluster:
    labels_: list

class TestClustering:
    """Cluster id is determined by the first dimension of the vector being zero."""
    def fit(self, X):
        return TestCluster(X[:, 0] == 0)


class TestAgglomerativeVectorId(unittest.TestCase):
    def test_empty(self):
        vector_id = ClusterIdentity(TestClustering(), 3)
        self.assertEqual((0, 3), vector_id._representations.shape)

    def test_add_one_dimensional(self):
        vector_id = ClusterIdentity(TestClustering(), 3)

        vector_id.add(np.array([0,3,4]))
        vector_id.add(np.array([0,3,4]))

        np.testing.assert_array_almost_equal([[0,3,4], [0,3,4]], vector_id._representations)
        np.testing.assert_array_equal([0,3/5,4/5], next(iter(vector_id._centroids.values())))

    def test_add_first(self):
        vector_id = ClusterIdentity(TestClustering(), 3)

        vector_id.add(np.array([[0,3,4]]))

        np.testing.assert_array_almost_equal([[0,3,4]], vector_id._representations)
        np.testing.assert_array_equal([0,3/5,4/5], next(iter(vector_id._centroids.values())))

    def test_add_one_cluster(self):
        vector_id = ClusterIdentity(TestClustering(), 3)

        vector_id.add(np.array([[0,3,4], [0,3,4]]))

        np.testing.assert_array_equal([[0,3,4], [0,3,4]], vector_id._representations)
        np.testing.assert_array_almost_equal([0,3/5,4/5], next(iter(vector_id._centroids.values())))

    def test_add_two_clusters(self):
        vector_id = ClusterIdentity(TestClustering(), 3)

        vector_id.add(np.array([[0,3,4], [0,3,4], [5,12,0], [5,12,0]]))

        np.testing.assert_array_equal([[0,3,4], [0,3,4], [5,12,0], [5,12,0]], vector_id._representations)

        centroids = list(sorted([tuple(cent) for cent in vector_id._centroids.values()]))
        self.assertEqual(2, len(centroids))
        np.testing.assert_array_almost_equal([0,3/5,4/5], centroids[0])
        np.testing.assert_array_almost_equal([5/13,12/13,0], centroids[1])

    def test_add_two_clusters_one_by_one(self):
        vector_id = ClusterIdentity(TestClustering(), 3)

        id_1 = vector_id.add(np.array([[0,3,4]]))
        id_2 = vector_id.add(np.array([[0,3,4]]))
        id_3 = vector_id.add(np.array([[5,12,0]]))
        id_4 = vector_id.add(np.array([[5,12,0]]))

        np.testing.assert_array_equal([[0,3,4], [0,3,4], [5,12,0], [5,12,0]], vector_id._representations)

        centroids = list(sorted([tuple(cent) for cent in vector_id._centroids.values()]))
        self.assertEqual(2, len(centroids))
        np.testing.assert_array_almost_equal([0,3/5,4/5], centroids[0])
        np.testing.assert_array_almost_equal([5/13,12/13,0], centroids[1])

        self.assertEqual(id_1, id_2)
        self.assertEqual(id_3, id_4)
        self.assertNotEqual(id_1, id_3)

    def test_add_two_clusters_agglomerative(self):
        vector_id = ClusterIdentity.agglomerative(3, 0.1)

        id_1 = vector_id.add(np.array([[0,3,4]]))
        id_2 = vector_id.add(np.array([[0,3,4]]))
        id_3 = vector_id.add(np.array([[5,12,0]]))
        id_4 = vector_id.add(np.array([[5,12,0]]))

        np.testing.assert_array_equal([[0,3,4], [0,3,4], [5,12,0], [5,12,0]], vector_id._representations)

        centroids = list(sorted([tuple(cent) for cent in vector_id._centroids.values()]))
        self.assertEqual(2, len(centroids))
        np.testing.assert_array_almost_equal([0,3/5,4/5], centroids[0])
        np.testing.assert_array_almost_equal([5/13,12/13,0], centroids[1])

        self.assertEqual(id_1, id_2)
        self.assertEqual(id_3, id_4)
        self.assertNotEqual(id_1, id_3)

    def test_add_one_dimensional_no_dim(self):
        vector_id = ClusterIdentity(TestClustering(), 0)

        vector_id.add(np.array([0,3,4]))
        vector_id.add(np.array([0,3,4]))

        np.testing.assert_array_almost_equal([[0,3,4], [0,3,4]], vector_id._representations)
        np.testing.assert_array_equal([0,3/5,4/5], next(iter(vector_id._centroids.values())))

    def test_two_dimensional_no_dim(self):
        vector_id = ClusterIdentity(TestClustering(), None)

        vector_id.add(np.array([[0,3,4]]))
        vector_id.add(np.array([[0,3,4]]))

        np.testing.assert_array_almost_equal([[0,3,4], [0,3,4]], vector_id._representations)
        np.testing.assert_array_equal([0,3/5,4/5], next(iter(vector_id._centroids.values())))