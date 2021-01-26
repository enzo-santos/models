import unittest

from models.clustering.cluster import _Cluster
from models.utils.linear_alg import Vector


class ClusterTestCase(unittest.TestCase):
    def test_Init(self):
        cluster: _Cluster = _Cluster([0, 0])
        self.assertEqual([0, 0], cluster.center)

    def test_Iterable(self):
        cluster: _Cluster = _Cluster([0, 0])
        self.assertEqual(0, len(cluster))
        for _ in cluster:
            self.assertTrue(False)

        cluster.add([1, 2])
        self.assertEqual(1, len(cluster))
        for point in cluster:
            self.assertEqual([1, 2], point)

        cluster = _Cluster([1, 2], [[2, 3], [3, 2]])
        self.assertEqual(2, len(cluster))
        iter_ = iter(cluster)
        self.assertEqual([2, 3], next(iter_))
        self.assertEqual([3, 2], next(iter_))
        self.assertRaises(StopIteration, lambda: next(iter_))

        cluster.add([3, 3])
        self.assertEqual(3, len(cluster))

    def test_TotalDistance(self):
        cluster: _Cluster = _Cluster([0, 0])
        self.assertEqual(0, cluster.distance())

        cluster.add([0, 0])
        self.assertEqual(0, cluster.distance())

        cluster.add([3, 4])
        self.assertEqual(5, cluster.distance())

        cluster.add([5, 12])
        self.assertEqual(18, cluster.distance())

    def test_Repr(self):
        center: Vector = [0, 0]
        cluster: _Cluster = _Cluster(center)
        repr_ = repr(cluster)
        self.assertIn("_Cluster", repr_)
        self.assertIn(f"{center}", repr_)

        sample: Vector = [1, 2]
        cluster.add(sample)
        repr_ = repr(cluster)
        self.assertIn("_Cluster", repr_)
        self.assertIn(f"{center}", repr_)
        self.assertIn(f"{sample}", repr_)
