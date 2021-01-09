import random
from typing import Tuple, Union

from models.utils.linear_alg import Matrix, Vector, check_dimension_match, diagonal_sum
from models.utils.lists import split_list


def split_into_train_test(
        X: Matrix,
        y: Vector,
        test_sampling=0.3,
) -> Tuple[Matrix, Vector, Matrix, Vector]:
    """
    Divide amostras em conjuntos aleatórios de treinamento e teste.

    Equivalente à sklearn.model_selection.train_test_split_.

    :param X: matriz que representa as características de cada amostra.
    :param y: vetor que representa as classes de cada amostra.
    :param test_sampling: proporção do conjunto de dados original a ser incluído nas amostras de teste a serem
                            retornadas, padrão 30%.
    :return: conjuntos resultantes da divisão das amostras de entrada.

    .. _sklearn.model_selection.train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    n_samples = check_dimension_match(X, y)
    list_samples = [(f, c) for f, c in zip(X, y)]
    shuffled_samples = random.sample(list_samples, n_samples)

    train_size = int(n_samples * (1 - test_sampling))
    train_samples = shuffled_samples[:train_size]
    test_samples = shuffled_samples[train_size:]

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for X_, y_ in train_samples:
        X_train.append(X_)
        y_train.append(y_)

    for X_, y_ in test_samples:
        X_test.append(X_)
        y_test.append(y_)

    return X_train, y_train, X_test, y_test


def confusion_matrix(
        y_true: Vector,
        y_pred: Vector,
        n_classes: int = None,
        return_classes: bool = False,
) -> Union[Matrix, Tuple[Matrix, Vector]]:
    """
    Calcula a matriz de confusão para avaliar a precisão de uma classificação.

    Equivalente à sklearn.metrics.confusion_matrix_.

    :param y_true: valores alvo corretos.
    :param y_pred: alvos estimados conforme retornados por um classificador.
    :param n_classes: o número de classes usadas na classificação. Se `None`, aqueles que aparecem pelo menos uma vez
                        em 'y_true' e 'y_pred' são usados.
    :param return_classes: se as classes usadas pela matriz de confusão devem ser retornadas. Apenas as classes que
                            aparecem pelo menos uma vez em 'y_true' e 'y_pred' serão retornadas.

    :return: a matriz de confusão. Se 'return_classes' for `True`, uma tupla será retornada com o segundo valor sendo
                uma lista das classes usadas pela matriz de confusão, onde cada classe corresponde a uma coluna da
                matriz.

    .. _sklearn.metrics.confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    classes = list(set(y_true) | set(y_pred))
    if n_classes is None:
        n_classes = len(classes)

    matrix = []
    for c in classes:
        pairs = [pair for pair in zip(y_pred, y_true) if pair[1] == c]
        pred_values = [t[0] for t in pairs]

        row = [0 for _ in range(n_classes)]
        for pred_value in pred_values:
            index = classes.index(pred_value)
            row[index] += 1
        matrix.append(row)

    # Adiciona classes que estejam faltando, caso tenham
    while len(matrix) < n_classes:
        matrix.append([])
        for row in matrix:
            row.extend([0] * (n_classes - len(row)))

    return (matrix, classes) if return_classes else matrix


def k_fold(clf, X: Matrix, y: Vector, *, n_folds: int = 5) -> Vector:
    """
    Utiliza o método de validação cruzada k-fold em amostras.

    Cada fold é usada como um conjunto de validação uma vez, enquanto os k-1 folds restantes formam o conjunto de
    treinamento.

    Equivalente à sklearn.model_selection.cross_val_score_ com o paramêtro 'cv' igual a 'n_folds'.

    :param clf: classificador a ser avaliado.
    :param X: matriz que representa as características de cada amostra.
    :param y: vetor que representa as classes de cada amostra.
    :param n_folds: número de folds cuja população total será dividida.

    :return: vetor de precisão resultante da classificação de cada iteração.

    .. _sklearn.model_selection.cross_val_score: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
    """
    n_samples = check_dimension_match(X, y, allow_empty=False)
    samples = [(f, c) for f, c in zip(X, y)]

    if n_folds < 2:
        raise ValueError(f"n_folds ({n_folds}) must be greater than one")

    if n_folds > n_samples:
        raise ValueError(f"n_folds ({n_folds}) must be less or equal than the number of samples ({n_samples})")

    accuracies = []
    for _ in range(n_folds):
        shuffled_samples = random.sample(samples, len(samples))
        folds = split_list(shuffled_samples, n_folds)

        folds_out, folds_in = folds[:1], folds[1:]

        X_train = []
        y_train = []
        for fold in folds_in:
            for X_, y_ in fold:
                X_train.append(X_)
                y_train.append(y_)

        X_test = []
        y_test = []
        for fold in folds_out:
            for X_, y_ in fold:
                X_test.append(X_)
                y_test.append(y_)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        n_test = sum(sum(x) for x in cm)
        accuracy = diagonal_sum(cm) / n_test

        accuracies.append(accuracy)

    return accuracies
