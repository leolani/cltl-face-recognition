import dataclasses
import unittest

import numpy as np

from cltl.vector_id.clusterid import ClusterIdentity

@dataclasses.dataclass
class TestCluster:
    labels_: list

class TestClustering:
    def fit(self, X):
        return TestCluster(X[:, 0].ravel())


class TestAgglomerativeVectorId(unittest.TestCase):
    def test_empty(self):
        vector_id = ClusterIdentity(TestClustering(), 3, 0.01)
        self.assertEqual((0, 3), vector_id._representations.shape)

    def test_add(self):
        vector_id = ClusterIdentity(TestClustering(), 3, 0.01)

        vector_id.add(np.array([[0,1,2], [0,1,2]]))

        np.testing.assert_array_equal([[0,1,2], [0,1,2]], vector_id._representations)
        np.testing.assert_array_equal([0,1,2], next(iter(vector_id._centroids.values())))


