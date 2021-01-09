import unittest

from models.clustering.cluster import Cluster
from models.clustering.datapoint import DataPoint
from models.clustering.position import Position


class ClusterTestCase(unittest.TestCase):
    def test_Init(self):
        cluster: Cluster = Cluster(Position(0, 0))
        self.assertEqual(Position(0, 0), cluster.position)

    def test_Iterable(self):
        cluster: Cluster = Cluster(Position(0, 0))
        self.assertEqual(0, len(cluster))
        for _ in cluster:
            self.assertTrue(False)

        cluster.add(DataPoint("a", Position(1, 2)))
        self.assertEqual(1, len(cluster))
        for point in cluster:
            self.assertEqual(DataPoint("a", Position(1, 2)), point)

        cluster = Cluster(Position(1, 2), [DataPoint("b", Position(2, 3)), DataPoint("c", Position(3, 2))])
        self.assertEqual(2, len(cluster))
        iter_ = iter(cluster)
        self.assertEqual(DataPoint("b", Position(2, 3)), next(iter_))
        self.assertEqual(DataPoint("c", Position(3, 2)), next(iter_))
        self.assertRaises(StopIteration, lambda: next(iter_))

        cluster.add(DataPoint("d", Position(3, 3)))
        self.assertEqual(3, len(cluster))

    def test_TotalDistance(self):
        cluster: Cluster = Cluster(Position(0, 0))
        self.assertEqual(0, cluster.distance())

        cluster.add(DataPoint("p0", Position(0, 0)))
        self.assertEqual(0, cluster.distance())

        cluster.add(DataPoint("p1", Position(3, 4)))
        self.assertEqual(5, cluster.distance())

        cluster.add(DataPoint("p2", Position(5, 12)))
        self.assertEqual(18, cluster.distance())

    def test_Repr(self):
        position: Position = Position(0, 0)
        cluster: Cluster = Cluster(position)
        repr_ = repr(cluster)
        self.assertIn("Cluster", repr_)
        self.assertIn(f"{position}", repr_)

        position = Position(1, 2)
        point: DataPoint = DataPoint("p0", position)
        cluster.add(point)
        repr_ = repr(cluster)
        self.assertIn("Cluster", repr_)
        self.assertIn(f"{position}", repr_)
        self.assertIn(f"{point}", repr_)
