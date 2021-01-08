import unittest

from classifiers.clustering.position import Position


class PositionTestCase(unittest.TestCase):
    def test_instantiation(self):
        position: Position = Position(1, 2)
        self.assertEqual(1, position.x)
        self.assertEqual(2, position.y)

        position = Position(0, 0)
        self.assertEqual(position.x, position.y)

    def test_distance(self):
        p0: Position = Position(0, 3)
        p1: Position = Position(4, 0)
        self.assertEqual(5, p0.distance_to(p1))

        p0 = Position(10, 10)
        self.assertEqual(0, p0.distance_to(p0))

        p0 = Position(1, 2)
        p1 = Position(3, 4)
        self.assertEqual(p0.distance_to(p1), p1.distance_to(p0))

        d0: float = Position(5, 0).distance_to(Position(0, 12))
        d1: float = Position(0, 5).distance_to(Position(12, 0))
        self.assertEqual(d0, d1)

        d0 = Position(5, 6).distance_to(Position(7, 8))
        d1 = Position(0, 10).distance_to(Position(1, 12))
        self.assertNotEqual(d0, d1)

    def test_equality(self):
        p0: Position = Position(1, 2)
        self.assertEqual(p0, p0)
        self.assertNotEqual(p0, (1, 2))

        p1: Position = Position(3, 4)
        self.assertNotEqual(p0, p1)

        p1 = Position(2, 1)
        self.assertNotEqual(p0, p1)

        self.assertEqual(hash(p0), hash(p0))

        p1 = Position(3, 4)
        self.assertNotEqual(hash(p0), hash(p1))

        p1 = Position(2, 1)
        self.assertNotEqual(hash(p0), hash(p1))

    def test_repr(self):
        p0: Position = Position(1, 2)
        repr_ = repr(p0)
        self.assertIn(f"Position", repr_)
        self.assertIn(f"{p0.x}", repr_)
        self.assertIn(f"{p0.y}", repr_)
