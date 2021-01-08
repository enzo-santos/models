import collections
from typing import List

from models.clustering.datapoint import DataPoint
from models.clustering.position import Position


class Cluster(collections.abc.Iterable, collections.abc.Sized):
    """
    Representa um cluster que agrupa pontos.
    """

    def __init__(self, position: Position, points: List[DataPoint] = None):
        """
        Cria um cluster.

        :param position: a posição do centróide desse cluster.
        :param points: os pontos desse cluster. Se `None`, o cluster está vazio.
        """
        self.position: Position = position
        self._children: List[DataPoint] = [] if points is None else points

    def add(self, point: DataPoint):
        """
        Adiciona um ponto a este cluster.

        :param point: o ponto a ser adicionado.
        """
        self._children.append(point)

    def distance(self):
        """
        :return: a soma das distâncias do centroide do cluster a todos os seus pontos.
        """
        return sum(self.position.distance_to(point.position) for point in self._children)

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        return len(self._children)

    def __repr__(self):
        return f'Cluster({self.position}, {self._children})'
