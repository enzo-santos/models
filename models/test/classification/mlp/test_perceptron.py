import unittest

from models.classification.mlp.gradient import Gradient
from models.classification.mlp.perceptron import Perceptron, InputPerceptron, UpdatingPerceptron
from models.utils.math import sigmoid


class PerceptronTestCase(unittest.TestCase):
    def test_Init(self):
        perceptron: Perceptron

        perceptron = Perceptron(input_size=3, bias=0.2)
        self.assertEqual(3, len(perceptron.weights))
        self.assertEqual(0.2, perceptron.bias)
        self.assertIsNone(perceptron.last_input)
        self.assertIsNone(perceptron.last_output)

        perceptron = Perceptron(input_size=4, bias=-10.0, weights=[0.1, 1.2, 2.3, 3.4])
        self.assertListEqual([0.1, 1.2, 2.3, 3.4], perceptron.weights)
        self.assertEqual(-10.0, perceptron.bias)
        self.assertIsNone(perceptron.last_input)
        self.assertIsNone(perceptron.last_output)

        self.assertRaises(ValueError, lambda: Perceptron(input_size=2, bias=0.0, weights=[0, 1, 2]))
        self.assertRaises(ValueError, lambda: Perceptron(input_size=4, bias=0.0, weights=[0, 1, 2]))

    def test_Transform(self):
        perceptron: Perceptron = Perceptron(input_size=3, bias=1.0)
        self.assertEqual(0.0, perceptron.transform(float('-inf')))
        self.assertEqual(0.5, perceptron.transform(0))
        self.assertEqual(1.0, perceptron.transform(float('+inf')))

    def test_Update(self):
        perceptron: Perceptron = Perceptron(input_size=4, bias=1e+2, weights=[1.0, 2.0, 3.0, 4.0])
        self.assertListEqual([1.0, 2.0, 3.0, 4.0], perceptron.weights)

        self.assertRaises(ValueError, lambda: perceptron.update([1.0, 2.0]))
        self.assertListEqual([1.0, 2.0, 3.0, 4.0], perceptron.weights)

        perceptron.update([5.0, 6.0, 7.0, 8.0])
        self.assertListEqual([5.0, 6.0, 7.0, 8.0], perceptron.weights)

    def test_Predict(self):
        perceptron: Perceptron = Perceptron(input_size=3, bias=1.0, weights=[2.0, 3.0, 5.0])
        self.assertRaises(ValueError, lambda: perceptron.predict([4.0, 5.0]))

        # 1.0 + 0.5 * 2.0 + 0.2 * 3.0 + 0.1 * 5.0 == 3.1
        expected: float = sigmoid(3.1)
        self.assertEqual(expected, perceptron.predict([0.5, 0.2, 0.1]))
        self.assertEqual([0.5, 0.2, 0.1], perceptron.last_input)
        self.assertEqual(expected, perceptron.last_output)


class InputPerceptronTestCase(unittest.TestCase):
    def test_Init(self):
        perceptron: InputPerceptron = InputPerceptron()
        self.assertEqual(1, len(perceptron.weights))
        self.assertListEqual([1.0], perceptron.weights)
        self.assertEqual(0.0, perceptron.bias)
        self.assertIsNone(perceptron.last_input)
        self.assertIsNone(perceptron.last_output)

    def test_Transform(self):
        perceptron: InputPerceptron = InputPerceptron()
        self.assertEqual(0.0, perceptron.transform(0.0))
        self.assertEqual(0.5, perceptron.transform(0.5))
        self.assertEqual(1.0, perceptron.transform(1.0))

    def test_Predict(self):
        perceptron: InputPerceptron = InputPerceptron()
        self.assertRaises(ValueError, lambda: perceptron.predict([2.0, 2.0]))

        self.assertEqual(3.0, perceptron.predict([3.0]))
        self.assertEqual([3.0], perceptron.last_input)
        self.assertEqual([perceptron.last_output], perceptron.last_input)


class UpdatingPerceptronTestCase(unittest.TestCase):
    def test_Init(self):
        perceptron: UpdatingPerceptron

        perceptron = UpdatingPerceptron(Perceptron(input_size=3, bias=0.2))
        self.assertEqual(3, len(perceptron.weights))
        self.assertEqual(0.2, perceptron.bias)
        self.assertIsNone(perceptron.last_input)
        self.assertIsNone(perceptron.last_output)

        perceptron = UpdatingPerceptron(Perceptron(input_size=4, bias=-10.0, weights=[0.1, 1.2, 2.3, 3.4]))
        self.assertListEqual([0.1, 1.2, 2.3, 3.4], perceptron.weights)
        self.assertEqual(-10.0, perceptron.bias)
        self.assertIsNone(perceptron.last_input)
        self.assertIsNone(perceptron.last_output)

        self.assertRaises(ValueError, lambda: UpdatingPerceptron(Perceptron(input_size=2, bias=0.0, weights=[0, 1, 2])))
        self.assertRaises(ValueError, lambda: UpdatingPerceptron(Perceptron(input_size=4, bias=0.0, weights=[0, 1, 2])))

    def test_Put(self):
        perceptron = UpdatingPerceptron(Perceptron(input_size=3, bias=0.2))
        self.assertListEqual([], perceptron.gradients)
        perceptron.put(Gradient())
        self.assertListEqual([Gradient()], perceptron.gradients)
        perceptron.put(Gradient(δa=0.4))
        perceptron.put(Gradient(δw=0.6))
        self.assertListEqual([Gradient(), Gradient(δa=0.4), Gradient(δw=0.6)], perceptron.gradients)
        perceptron.put(Gradient(δa=0.9, δw=1.9))
        self.assertListEqual([Gradient(), Gradient(δa=0.4), Gradient(δw=0.6), Gradient(δa=0.9, δw=1.9)],
                             perceptron.gradients)

    def test_Update(self):
        p0: Perceptron = Perceptron(input_size=4, bias=1e+2, weights=[1.0, 2.0, 3.0, 4.0])

        perceptron: UpdatingPerceptron
        perceptron = UpdatingPerceptron(p0)
        self.assertListEqual([1.0, 2.0, 3.0, 4.0], p0.weights)

        self.assertRaises(ValueError, lambda: perceptron.update([1.0, 2.0]))
        self.assertListEqual([1.0, 2.0, 3.0, 4.0], p0.weights)

        perceptron.update([5.0, 6.0, 7.0, 8.0])
        self.assertListEqual([5.0, 6.0, 7.0, 8.0], p0.weights)


if __name__ == '__main__':
    unittest.main()
