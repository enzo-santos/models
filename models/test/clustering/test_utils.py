import unittest

from models.clustering.utils import distance


class UtilsTestCase(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(5, distance((0, 3), (4, 0)))
        self.assertEqual(distance((1, 2), (3, 4)), distance((3, 4), (1, 2)))
        self.assertEqual(distance((5, 0), (0, 12)), distance((0, 5), (12, 0)))
        self.assertEqual(0, distance((10, 10), (10, 10)))
        self.assertNotEqual(distance((5, 6), (7, 8)), distance((0, 10), (1, 12)))
