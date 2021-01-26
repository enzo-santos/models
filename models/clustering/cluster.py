from typing import List, Iterable, Sized

from models.utils.linear_alg import Vector, euclidean


class _Cluster(Iterable[Vector], Sized):
    """
    Representa um cluster que agrupa amostras.
    """

    def __init__(self, center: Vector, samples: List[Vector] = None):
        """
        Cria um cluster.

        :param center: a amostra que representa o centroide desse cluster.
        :param samples: as amostras desse cluster. Se `None`, o cluster está vazio.
        """
        self.center: Vector = center
        self._samples: List[Vector] = [] if samples is None else samples

    def add(self, sample: Vector):
        """
        Adiciona uma amostra a este cluster.

        :param sample: a amostra a ser adicionada.
        """
        self._samples.append(sample)

    def distance(self):
        """
        :return: a soma das distâncias do centroide do cluster a todos os seus pontos.
        """
        return sum(euclidean(self.center, sample) for sample in self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __len__(self):
        return len(self._samples)

    def __repr__(self):
        return f'_Cluster({self.center}, {self._samples})'
