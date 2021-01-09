import unittest
import unittest.mock
from typing import List, Sequence

from models.classification.utils import split_into_train_test, confusion_matrix, k_fold
from models.utils.linear_alg import Vector, Matrix


class _MockClassifier:
    @classmethod
    def empty(cls):
        return cls(())

    def __init__(self, y_preds: Sequence[Vector]):
        self._iter = iter(y_preds)

    def fit(self, X: Matrix, y: Vector) -> '_MockClassifier':
        return self

    def predict(self, _: Matrix) -> Vector:
        return next(self._iter)


class UtilsTestCase(unittest.TestCase):
    @staticmethod
    def _mock_ListShuffle(values: List, n: int):
        return values[:n]

    def test_SplitIntoTrainTest(self):
        with unittest.mock.patch('random.sample', self._mock_ListShuffle):
            result = split_into_train_test(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                [0, 1, 2, 3],
                test_sampling=0.5,
            )
            X_train, y_train, X_test, y_test = result
            self.assertEqual([[1, 2, 3], [4, 5, 6]], X_train)
            self.assertEqual([0, 1], y_train)
            self.assertEqual([[7, 8, 9], [10, 11, 12]], X_test)
            self.assertEqual([2, 3], y_test)

            result = split_into_train_test(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                [0, 1, 2, 3],
                test_sampling=0.25,
            )
            X_train, y_train, X_test, y_test = result
            self.assertEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9]], X_train)
            self.assertEqual([0, 1, 2], y_train)
            self.assertEqual([[10, 11, 12]], X_test)
            self.assertEqual([3], y_test)

            result = split_into_train_test(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [0, 1, 2],
                test_sampling=0.33,
            )
            X_train, y_train, X_test, y_test = result
            self.assertEqual([[1, 2, 3], [4, 5, 6]], X_train)
            self.assertEqual([0, 1], y_train)
            self.assertEqual([[7, 8, 9]], X_test)
            self.assertEqual([2], y_test)

    def test_ConfusionMatrix(self):
        self.assertEqual(
            [[1, 1],
             [1, 1]],
            confusion_matrix(
                [0, 0, 1, 1],
                [0, 1, 0, 1]
            )
        )
        self.assertEqual(
            [[3, 2],
             [1, 4]],
            confusion_matrix(
                [0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
            ),
        )
        self.assertEqual(
            [[5]],
            confusion_matrix(
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            )
        )
        self.assertEqual(
            [[6, 0],
             [0, 0]],
            confusion_matrix(
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                n_classes=2,
            )
        )
        self.assertEqual(
            ([[1, 2],
              [3, 4]], [0, 1]),
            confusion_matrix(
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                return_classes=True,
            )
        )
        self.assertEqual(
            ([[2, 0, 0],
              [0, 0, 0],
              [0, 0, 0]], [1]),
            confusion_matrix(
                [1, 1],
                [1, 1],
                n_classes=3,
                return_classes=True
            )
        )

    def test_KFold(self):
        clf = _MockClassifier.empty()
        self.assertRaises(ValueError, lambda: k_fold(clf, [], []))
        self.assertRaises(ValueError, lambda: k_fold(clf, [[1, 2], [3, 4]], [1, 2], n_folds=3))
        self.assertRaises(ValueError, lambda: k_fold(clf, [[1, 2, 3]], [1], n_folds=1))

        with unittest.mock.patch('random.sample', self._mock_ListShuffle):
            self.assertEqual(
                [1, 0.5],
                k_fold(
                    _MockClassifier((
                        [0, 1, 0, 1],  # 100% de acurácia
                        [1, 0, 0, 1],  # 50% de acurácia
                    )),
                    [
                        [1, 2, 3],
                        [1, 4, 7],
                        [4, 5, 6],
                        [2, 5, 8],
                        [7, 8, 9],
                        [3, 6, 9],
                        [1, 4, 6],
                        [1, 5, 8],
                    ],
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    n_folds=2,
                ),
            )

            self.assertEqual(
                [0.25, 0.5, 0.75],
                k_fold(
                    _MockClassifier((
                        [0, 0, 1, 0],  # 25% de acurácia
                        [0, 1, 1, 0],  # 50% de acurácia
                        [0, 1, 0, 0],  # 75% de acurácia
                    )),
                    [
                        [0, 1, 2, 3],
                        [1, 3, 5, 7],
                        [2, 5, 8, 9],
                        [0, 3, 8, 7],
                        [1, 5, 5, 3],
                        [2, 3, 2, 7],
                        [3, 5, 5, 0],
                        [4, 5, 2, 3],
                        [5, 4, 3, 2],
                        [3, 2, 1, 0],
                        [4, 4, 1, 1],
                        [4, 2, 3, 3],
                    ],
                    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    n_folds=3,
                ),
            )
