import unittest

from models.classification.mlp.model import MLPClassifier
from models.utils.linear_alg import Matrix, Vector


class ModelTestCase(unittest.TestCase):
    def test_Init(self):
        clf: MLPClassifier = MLPClassifier(n_layers=2, layer_size=3, learning_rate=0.1, n_generations=10)
        self.assertEqual(2, clf.n_layers)
        self.assertEqual(3, clf.layer_size)
        self.assertEqual(10, clf.n_generations)
        self.assertEqual(0.1, clf.learning_rate)
        self.assertIsNone(clf.layers)

    def test_Fit(self):
        clf: MLPClassifier
        X: Matrix = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y: Vector = [0, 0, 1, 1]

        clf = MLPClassifier(n_layers=3, layer_size=4)

        self.assertIsNone(clf.layers)
        self.assertRaises(ValueError, lambda: clf.fit(X, [0, 1, 1, 2]))

        self.assertIsNone(clf.layers)
        clf.fit(X, y)
        self.assertEqual(5, len(clf.layers))
        self.assertListEqual([2, 4, 4, 4, 1], [len(layer) for layer in clf.layers])

        clf = MLPClassifier(n_layers=1, layer_size=5)
        self.assertIsNone(clf.layers)
        clf.fit(X, y)
        self.assertEqual(3, len(clf.layers))
        self.assertListEqual([2, 5, 1], [len(layer) for layer in clf.layers])

        clf = MLPClassifier(n_layers=0, layer_size=3)
        self.assertIsNone(clf.layers)
        clf.fit(X, y)
        self.assertEqual(2, len(clf.layers))
        self.assertListEqual([2, 1], [len(layer) for layer in clf.layers])

    def test_Predict(self):
        X: Matrix = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y: Vector = [0, 0, 0, 1]

        clf: MLPClassifier = MLPClassifier(n_layers=1, layer_size=4, n_generations=1_000)
        self.assertRaises(ValueError, lambda: clf.predict([1, 0]))

        clf.fit(X, y)
        self.assertEqual(4, len(clf.predict(X)))
        self.assertIsInstance(clf.predict([0, 0]), int)

    def test_XorPrediction(self):
        X: Matrix = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y: Vector = [0, 1, 1, 1]

        clf: MLPClassifier = MLPClassifier(n_layers=1, layer_size=4, learning_rate=0.7, n_generations=1_000)
        clf.fit(X, y)

        for sample, o_true in zip(X, y):
            value, = clf.transform(sample)
            o_pred = clf.predict(sample)
            self.assertEqual(o_true, o_pred)

            error = ((o_true - value) ** 2.0) / 2.0
            self.assertLessEqual(error, 1e-1)


if __name__ == '__main__':
    unittest.main()
