from typing import List, TypeVar, Any

T = TypeVar('T')
R = TypeVar('R')


def rows(matrix: List[List[Any]]):
    """
    Retorna o número de linhas de uma matriz.

    :param matrix: a matriz.
    :return: o número de linhas dessa matriz.
    """
    return len(matrix)


def cols(matrix: List[List[Any]], *, check_if_equal=True):
    """
    Retorna o número de colunas de uma matriz.

    :param matrix: a matriz.
    :param check_if_equal: se é obrigatório as linhas dessa matriz possuírem comprimentos iguais.
    :raises ValueError: se a matriz não possuir nenhuma linha ou, se ``check_if_equal`` for `True`, se as linhas dessa
                            matriz possuírem comprimentos diferentes.
    :return: o número de colunas dessa matriz.
    """
    if rows(matrix) == 0:
        raise ValueError("this matrix does not have any rows")

    v0 = matrix[0]
    if check_if_equal and not all(len(v0) == len(v) for v in matrix):
        raise ValueError("columns in this matrix are not the same size")

    return len(v0)


def split_list(array: List[T], n_lists: int) -> List[List[T]]:
    """
    Divide uma lista em um conjunto de listas.

    Equivalente à numpy.array_split_.

    :param array: lista a ser dividida.
    :param n_lists: número de listas a serem geradas a partir da lista original.
    :return: conjunto de listas geradas a partir da lista original.

    .. _numpy.array_split: https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
    """
    k, m = divmod(len(array), n_lists)

    if n_lists > len(array):
        raise ValueError(f"'n_lists' ({n_lists}) must be less \
                or equal than the length of the array ({len(array)})")

    lists = []
    for i in range(n_lists):
        p0 = i * k + min(i, m)
        p1 = (i + 1) * k + min(i + 1, m)
        lists.append(array[p0:p1])

    return lists


def sort_by(a1: List[T], a2: List[T], reverse: bool = False) -> List[T]:
    """
    Ordena uma lista com base nos valores de outra lista.

    :param a1: lista base.
    :param a2: lista cujos valores serão ordenados com base na ordenação de 'a1'.
    :param reverse: se a ordenação deve ser decrescente.

    :return: valores de 'a2' ordenados com base em 'a1'.
    """
    sorted_zip = sorted(zip(a1, a2), reverse=reverse, key=lambda t: t[0])
    return [t[1] for t in sorted_zip]
