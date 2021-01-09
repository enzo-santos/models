from typing import List

Matrix = List[List[float]]

Vector = List[float]


def check_dimension_match(X: Matrix, y: Vector, allow_empty: bool = True) -> int:
    """
    Verifica se uma matriz e um vetor possuem dimensões compatíveis.

    Uma matriz M e um vetor V possuem dimensões compatíveis se o número de linhas
    de M é o mesmo número de elementos de V.

    :param X: a matriz a ser verificada.
    :param y: o vetor a ser verificado.
    :param allow_empty: define se matrizes e vetores vazios são permitidos.
    :raises ValueError: se a matriz e o vetor não possuem dimensões compatíveis.
    :return: a dimensão da matriz e do vetor.
    """
    rows_X = len(X)
    size_y = len(y)
    if rows_X != size_y:
        raise ValueError(f"X rows ({rows_X}) must be the same as y size ({size_y})")

    dimension = rows_X
    if not allow_empty and not dimension:
        raise ValueError("X and y must not be empty")

    if dimension:
        len_row = len(X[0])
        if any(len(row) != len_row for row in X):
            raise ValueError("X rows must have the same length")

    return dimension


def diagonal_sum(matrix: Matrix) -> float:
    """
    Calcula a soma da diagonal principal de uma matriz.

    :param matrix: matriz quadrada cuja soma da diagonal será calculada.
    :raises ValueError: se a matriz não for quadrada ou se o número de linhas não forem iguais.
    :return: soma da diagonal principal.
    """
    if len(matrix) == 0:
        raise ValueError('the matrix must not be empty')
    if any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError('all the rows of the matrix must be the same length')
    if len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be square')

    return sum(row[i] for i, row in enumerate(matrix))


def minkowski(v1: Vector, v2: Vector, p: float) -> float:
    """
    Calcula a distância entre dois vetores usando a métrica Minkowski.

    A métrica Minkowski é considerada a generalização das métricas euclidiana e Manhattan.

    :param v1: o primeiro vetor.
    :param v2: o segundo vetor.
    :param p: a ordem da métrica.
    :return: a distância Minkowski entre os dois vetores.
    """
    return sum(abs(e1 - e2) ** p for e1, e2 in zip(v1, v2)) ** (1 / p)


def manhattan(v1: Vector, v2: Vector) -> float:
    """
    Calcula a distância entre dois vetores usando a métrica Manhattan.

    A distância Manhattan entre dois vetores é a soma das diferenças absolutas nas suas coordenadas cartesianas.

    :param v1: o primeiro vetor.
    :param v2: o segundo vetor.
    :return: a distância Manhattan entre os dois vetores.
    """
    return minkowski(v1, v2, 1)


def euclidean(v1: Vector, v2: Vector) -> float:
    """
    Calcula a distância entre dois vetores usando a métrica euclidiana.

    A distância euclidiana entre dois vetores é o comprimento do segmento de reta que os conecta.

    :param v1: o primeiro vetor.
    :param v2: o segundo vetor.
    :return: a distância Euclidean entre os dois vetores.
    """
    return minkowski(v1, v2, 2)


def chebyshev(v1: Vector, v2: Vector) -> float:
    """
    Calcula a distância entre dois vetores usando a métrica Chebyshev.

    A distância Chebyshev entre dois vetores é a maior de suas diferenças em qualquer coordenada.

    :param v1: o primeiro vetor.
    :param v2: o segundo vetor.
    :return: a distância Chebyshev entre os dois vetores.
    """
    return max(abs(e1 - e2) for e1, e2 in zip(v1, v2))


def hamming(v1: Vector, v2: Vector) -> float:
    """
    Calcula a distância entre dois vetores usando a métrica Hamming.

    A distância Hamming entre dois vetores é o número de posições nas quais seus respectivos elementos em cada
    dimensão são diferentes.

    :param v1: o primeiro vetor.
    :param v2: o segundo vetor.
    :return: a distância Hamming entre os dois vetores.
    """
    return sum(e1 != e2 for e1, e2 in zip(v1, v2))
