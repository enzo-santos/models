import unittest
from typing import Callable

from models.utils.linear_alg import euclidean, diagonal_sum, manhattan, Vector, chebyshev, hamming, \
    check_dimension_match


class UtilsTestCase(unittest.TestCase):
    def test_CheckDimensionMatrix(self):
        self.assertEqual(0, check_dimension_match([], []))
        self.assertEqual(1, check_dimension_match([[1]], [2]))
        self.assertEqual(2, check_dimension_match([[1, 2], [3, 4]], [5, 6]))
        self.assertRaises(ValueError, lambda: check_dimension_match([[1], [2]], [3]))
        self.assertRaises(ValueError, lambda: check_dimension_match([], [], allow_empty=False))
        self.assertRaises(ValueError, lambda: check_dimension_match([[1, 2, 3], [4, 5]], [6, 7]))

    def test_DiagonalSum(self):
        self.assertEqual(0, diagonal_sum([[0]]))
        self.assertEqual(1, diagonal_sum([[1]]))
        self.assertEqual(5, diagonal_sum([[1, 2], [3, 4]]))
        self.assertEqual(15, diagonal_sum([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.assertRaises(ValueError, lambda: diagonal_sum([]))
        self.assertRaises(ValueError, lambda: diagonal_sum([[]]))
        self.assertRaises(ValueError, lambda: diagonal_sum([[1, 2], [3]]))
        self.assertRaises(ValueError, lambda: diagonal_sum([[1, 2]]))

    def _test_Distance(self, function: Callable[[Vector, Vector], float]):
        # Igual para pontos diferentes na mesma localização
        self.assertEqual(function([1, 2], [3, 4]), function([3, 4], [1, 2]))

        # Igual para coordenadas inversas
        self.assertEqual(function([5, 0], [0, 12]), function([0, 5], [12, 0]))

        # Zero para pontos iguais
        self.assertEqual(0, function([10, 10], [10, 10]))

    def test_EuclideanDistance(self):
        self._test_Distance(euclidean)
        self.assertEqual(5, euclidean([0, 3], [4, 0]))
        self.assertNotEqual(euclidean([5, 6], [7, 8]), euclidean([0, 10], [1, 12]))

    def test_ManhattanDistance(self):
        self._test_Distance(manhattan)
        self.assertEqual(7, manhattan([0, 3], [4, 0]))
        self.assertNotEqual(manhattan([5, 6], [7, 8]), manhattan([0, 10], [1, 12]))

    def test_ChebyshevDistance(self):
        self._test_Distance(chebyshev)
        self.assertEqual(2, chebyshev([5, 6], [7, 8]))
        self.assertNotEqual(chebyshev([1, 2], [3, 4]), chebyshev([1, 4], [9, 16]))

    def test_HammingDistance(self):
        self._test_Distance(hamming)
        self.assertEqual(2, hamming([1, 1], [2, 2]))
        self.assertNotEqual(hamming([5, 6], [5, 6]), hamming([10, 11], [10, 12]))
