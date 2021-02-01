import math


def sigmoid(x: float) -> float:
    """
    Calcula a função sigmoide.

    :param x: o valor de entrada.
    :return: o valor da função para o valor de entrada.
    """
    return 1.0 / (1.0 + math.exp(-x))
