import unittest

from models.utils.math import sigmoid


class MathTestCase(unittest.TestCase):
    def test_Sigmoid(self):
        self.assertEqual(0.0, sigmoid(float('-inf')))
        self.assertEqual(0.5, sigmoid(0))
        self.assertEqual(1.0, sigmoid(float('+inf')))


if __name__ == '__main__':
    unittest.main()
