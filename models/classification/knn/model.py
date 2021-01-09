import collections
import statistics
from typing import Callable, Optional, Set, Sequence, Union, Tuple, Dict

import models
from models.classification.utils import confusion_matrix
from models.utils.linear_alg import Vector, Matrix, diagonal_sum, euclidean
from models.utils.linear_alg import check_dimension_match
from models.utils.lists import sort_by

Metric = Callable[[Vector, Vector], float]


class KNearestClassifier:
    @classmethod
    def best(
            cls,
            X: Matrix,
            y: Vector,
            *,
            ks: Sequence[int],
            n_folds: int = 5,
            metric: Metric = euclidean,
            return_report: bool = False,
    ) -> Union[int, Tuple[int, Dict[str, float]]]:
        if len(ks) < 2:
            raise ValueError("you must pass more than one 'k' to evaluate")

        report: Dict[int, float] = {}
        for k in ks:
            clf = KNearestClassifier(k=k, metric=metric)
            # Manter o caminho models.classification.utils
            accuracies = models.classification.utils.k_fold(clf, X, y, n_folds=n_folds)
            report[k] = statistics.mean(accuracies)

        best_k = max(report.items(), key=lambda item: item[1])[0]
        return (best_k, report) if return_report else best_k

    def __init__(self, *, k: int = 5, metric: Metric = euclidean):
        self.k = k
        self.metric = metric
        self._X: Optional[Matrix] = None
        self._y: Optional[Vector] = None
        self._classes: Optional[Set[float]] = None

    def fit(self, X: Matrix, y: Vector) -> 'KNearestClassifier':
        check_dimension_match(X, y, allow_empty=False)
        self._X = X
        self._y = y
        self._classes = set(y)
        return self

    def predict(self, X: Union[Vector, Matrix]) -> Union[float, Vector]:
        if self._X is None:
            raise ValueError("you must call 'fit' before calling 'predict'")

        is_vector = all(isinstance(v, (float, int)) for v in X)
        X = [X] if is_vector else X

        y: Vector = []
        for feature in X:
            distances = [self.metric(feature, fit_feature) for fit_feature in self._X]
            ordered_classes = sort_by(distances, self._y)
            k_nearest_classes = ordered_classes[:self.k]
            winner_class = collections.Counter(k_nearest_classes).most_common(1)[0][0]
            if is_vector:
                return winner_class

            y.append(winner_class)

        return y

    def score(self, X: Matrix, y: Vector) -> float:
        if self._X is None:
            raise ValueError("you must call 'fit' before calling 'score'")

        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred, len(self._classes))
        return diagonal_sum(cm) / sum(sum(row) for row in cm)
