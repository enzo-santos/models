from classifiers.clustering.position import Position


class DataPoint:
    """
    Representa uma posição em um plano cartesiano que armazena um dado.
    """

    def __init__(self, id_: str, position: Position):
        """
        Cria um ponto de dado.

        :param id_: o dado que este ponto armazena.
        :param position: a posição deste ponto.
        """
        self.id = id_
        self.position = position

    def distance_to(self, other: Position) -> float:
        """
        Calcula a distância para outra coordenada.

        :param other: a outra coordenada.
        :return: a distância entre esta coordenada e a outra coordenada.
        """
        return self.position.distance_to(other)

    def __repr__(self):
        return f'DataPoint({self.id}, {self.position})'

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, DataPoint) and self.id == other.id
