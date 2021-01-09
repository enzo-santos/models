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
    """
    Classificador que implementa a votação de k-vizinhos mais próximos.
    """

    @staticmethod
    def best(
            X: Matrix,
            y: Vector,
            *,
            ks: Sequence[int],
            n_folds: int = 5,
            metric: Metric = euclidean,
            return_report: bool = False,
    ) -> Union[int, Tuple[int, Dict[str, float]]]:
        """
        Encontra um valor de k que possui a melhor acurácia para um conjunto de amostras.

        :param X: as características das amostras totais.
        :param y: as classes das amostras totais.
        :param ks: os valores de k a serem testados.
        :param n_folds: o número de folds a serem utilizados na validação cruzada, padrão 5.
        :param metric: a métrica de distância a ser utilizada, padrão euclidiana.
        :param return_report: define se deve-se retornar a acurácia de cada valor de k testado.
        :return: o valor de k que obteve a melhor acurácia para os valores de k fornecidos. Caso 'return_report',
                    um dicionário com cada chave sendo o valor de k testado e seu valor sua respectiva acurácia
                    será retornado.
        """
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
        """
        Cria um classificador de k-vizinhos mais próximos.

        :param k: o número de vizinhos mais próximos a serem considerados, padrão 5.
        :param metric: a métrica de distância a ser considerada, padrão euclidiana.
        """
        self.k = k
        self.metric = metric
        self._X: Optional[Matrix] = None
        self._y: Optional[Vector] = None
        self._classes: Optional[Set[float]] = None

    def fit(self, X: Matrix, y: Vector) -> 'KNearestClassifier':
        """
        Ajusta o classificador ao conjunto de dados de treinamento.

        :param X: as características das amostras de treinamento.
        :param y: as classes das amostras de treinamento.
        :return: o classificador ajustado.
        """
        check_dimension_match(X, y, allow_empty=False)
        self._X = X
        self._y = y
        self._classes = set(y)
        return self

    def predict(self, X: Union[Vector, Matrix]) -> Union[float, Vector]:
        """
        Prediz as classes dos dados fornecidos.

        :param X: as características das amostras a serem previstas.
        :return: a(s) classe(s) das amostras fornecidas.
        """
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
        """
        Retorna a acurácia do classificador para dados de teste.

        :param X: as características das amostras de teste.
        :param y: as classes das amostras de teste.
        :return: a acurácia de classificação, entre 0 e 1.
        """
        if self._X is None:
            raise ValueError("you must call 'fit' before calling 'score'")

        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred, len(self._classes))
        return diagonal_sum(cm) / sum(sum(row) for row in cm)
