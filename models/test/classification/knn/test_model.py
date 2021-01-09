import random
import unittest
import unittest.mock
from typing import Sequence

from models.classification.knn.model import KNearestClassifier
from models.utils.linear_alg import euclidean, hamming, chebyshev, Vector


class ModelTestCase(unittest.TestCase):
    class _MockKFold:
        def __init__(self, total_accuracies: Sequence[Vector]):
            self._iter = iter(total_accuracies)

        def __call__(self, *args, **kwargs) -> Vector:
            return next(self._iter)

    def test_Init(self):
        clf: KNearestClassifier = KNearestClassifier()
        self.assertEqual(5, clf.k)
        self.assertEqual(euclidean, clf.metric)

        clf = KNearestClassifier(k=7)
        self.assertEqual(7, clf.k)
        self.assertEqual(euclidean, clf.metric)

        clf = KNearestClassifier(metric=hamming)
        self.assertEqual(5, clf.k)
        self.assertEqual(hamming, clf.metric)

        clf = KNearestClassifier(k=2, metric=chebyshev)
        self.assertEqual(2, clf.k)
        self.assertEqual(chebyshev, clf.metric)

    def test_Prediction(self):
        clf: KNearestClassifier = KNearestClassifier(k=1)
        self.assertRaises(ValueError, lambda: clf.predict([[1, 2], [3, 4]]))

        clf.fit([[0, 0], [10, 10], [15, 15]], [0, 0, 1])
        y_pred = clf.predict([[0, 0], [10, 10], [15, 15]])
        self.assertEqual([0, 0, 1], y_pred)

        clf = KNearestClassifier(k=5)
        clf.fit([
            [6, 0],
            [8, 0],
            [7, 0],
            [5, 11],
            [5, 7],
            [8, 10],
            [6, 11],
            [8, 0],
            [7, 0],
            [9, 11]],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        )
        self.assertEqual(0, clf.predict([5, 12]))
        self.assertEqual(1, clf.predict([6.5, 0.5]))
        self.assertEqual(0, clf.predict([9, 11]))
        self.assertEqual(1, clf.predict([8, 0]))

    def test_Score(self):
        clf: KNearestClassifier = KNearestClassifier(k=1)
        self.assertRaises(ValueError, lambda: clf.score([[1, 2], [3, 4]], [0, 1]))

        clf.fit([[0, 0], [10, 10], [15, 15], [20, 20]], [0, 0, 1, 1])
        self.assertEqual(1, clf.score([[0, 0], [10, 10], [15, 15], [20, 20]], [0, 0, 1, 1]))
        self.assertEqual(0.75, clf.score([[0, 0], [0, 10], [15, 15], [15, 15]], [0, 0, 1, 0]))
        self.assertEqual(0.5, clf.score([[0, 0], [10, 10], [10, 10], [10, 10]], [0, 0, 1, 1]))
        self.assertEqual(0.25, clf.score([[0, 0], [0, 0], [0, 0], [0, 0]], [0, 1, 1, 1]))
        self.assertEqual(0, clf.score([[0, 0], [10, 10], [15, 15], [20, 20]], [1, 1, 0, 0]))

    def test_BestKSearch(self):
        self.assertRaises(ValueError, lambda: KNearestClassifier.best([[1, 2], [3, 4]], [0, 1], ks=(1,)))
        with unittest.mock.patch('models.classification.utils.k_fold', self._MockKFold([
            [random.uniform(0.8, 0.9) for _ in range(5)],  # Melhores acurácias
            [random.uniform(0.7, 0.8) for _ in range(5)],
            [random.uniform(0.6, 0.7) for _ in range(5)],
            [random.uniform(0.5, 0.6) for _ in range(5)],
            [random.uniform(0.4, 0.5) for _ in range(5)],
        ])):
            self.assertEqual(
                3,
                KNearestClassifier.best(
                    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                    [0, 1, 1, 0, 1],
                    ks=range(3, 7),
                )
            )

        with unittest.mock.patch('models.classification.utils.k_fold', self._MockKFold([
            [random.uniform(0.3, 0.4) for _ in range(5)],  # Melhores acurácias
            [random.uniform(0.6, 0.7) for _ in range(5)],
            [random.uniform(0.1, 0.2) for _ in range(5)],
        ])):
            self.assertEqual(
                1,
                KNearestClassifier.best(
                    [[1, 2], [3, 4]],
                    [0, 1],
                    n_folds=2,
                    ks=range(2),
                )
            )

        with unittest.mock.patch('models.classification.utils.k_fold', self._MockKFold([
            [0.5 for _ in range(5)],
            [0.2 for _ in range(5)],
            [1 for _ in range(5)],
        ])):
            self.assertEqual(
                (10, {3: 0.5, 6: 0.2, 10: 1}),
                KNearestClassifier.best(
                    [[1, 2], [3, 4], [5, 6], [7, 8]],
                    [0, 1, 2, 1],
                    n_folds=2,
                    ks=(3, 6, 10),
                    return_report=True,
                )
            )
