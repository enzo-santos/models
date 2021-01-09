from typing import List


def split_list(array: List, n_lists: int):
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


def sort_by(a1: List, a2: List, reverse: bool = False) -> List:
    """
    Ordena uma lista com base nos valores de outra lista.

    :param a1: lista base.
    :param a2: lista cujos valores serão ordenados com base na ordenação de 'a1'.
    :param reverse: se a ordenação deve ser decrescente.

    :return: valores de 'a2' ordenados com base em 'a1'.
    """
    sorted_zip = sorted(zip(a1, a2), reverse=reverse, key=lambda t: t[0])
    return [t[1] for t in sorted_zip]
