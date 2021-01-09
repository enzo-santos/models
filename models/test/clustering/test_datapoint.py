import unittest

from models.clustering.datapoint import DataPoint
from models.clustering.position import Position


class DataPointTestCase(unittest.TestCase):
    def test_Init(self):
        p0: DataPoint = DataPoint("p0", Position(1, 2))
        self.assertEqual("p0", p0.id)
        self.assertEqual(Position(1, 2), p0.position)

    def test_DistanceTo(self):
        p0: DataPoint = DataPoint("p0", Position(0, 3))
        p1: DataPoint = DataPoint("p1", Position(4, 0))
        self.assertEqual(5, p0.distance_to(p1.position))

    def test_EqualsTo(self):
        p0: DataPoint = DataPoint("p0", Position(1, 2))
        self.assertEqual(p0, p0)
        self.assertNotEqual(p0, (1, 2))

        p1: DataPoint = DataPoint("p1", Position(3, 4))
        self.assertNotEqual(p0, p1)

        p1 = DataPoint("p0", Position(3, 4))
        self.assertEqual(p0, p1)

        p1 = DataPoint("p1", Position(1, 2))
        self.assertNotEqual(p0, p1)

        self.assertEqual(hash(p0), hash(p0))

        p1 = DataPoint("p1", Position(3, 4))
        self.assertNotEqual(hash(p0), hash(p1))

        p1 = DataPoint("p0", Position(3, 4))
        self.assertEqual(hash(p0), hash(p1))

        p1 = DataPoint("p1", Position(1, 2))
        self.assertNotEqual(hash(p0), hash(p1))

    def test_Repr(self):
        point: DataPoint = DataPoint("p0", Position(1, 2))
        repr_ = repr(point)
        self.assertIn("DataPoint", repr_)
        self.assertIn(f"{point.position}", repr_)
        self.assertIn("p0", repr_)
