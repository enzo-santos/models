import math
from typing import Tuple


def distance(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    """
    Calcula a distância euclidiana entre duas coordenadas.

    :param p0: uma tupla representando uma coordenada (x, y).
    :param p1: uma tupla representando uma coordenada (x, y).
    :return: a distância euclidiana entre essas coordenadas.
    """
    x0, y0 = p0
    x1, y1 = p1

    dx = (x0 - x1) ** 2
    dy = (y0 - y1) ** 2
    return math.sqrt(dx + dy)
