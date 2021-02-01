import unittest

from models.classification.mlp.gradient import Gradient


class GradientTestCase(unittest.TestCase):
    def test_Init(self):
        gradient: Gradient
        gradient = Gradient()
        self.assertIsNone(gradient.δa)
        self.assertIsNone(gradient.δw)

        gradient = Gradient(δa=0.5)
        self.assertEqual(0.5, gradient.δa)
        self.assertIsNone(gradient.δw)

        gradient = Gradient(δw=0.7)
        self.assertIsNone(gradient.δa)
        self.assertEqual(0.7, gradient.δw)

        gradient = Gradient(δa=1.0, δw=2.0)
        self.assertEqual(1.0, gradient.δa)
        self.assertEqual(2.0, gradient.δw)



if __name__ == '__main__':
    unittest.main()
