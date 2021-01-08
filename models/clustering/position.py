from models.clustering.utils import distance


class Position:
    """
    Representa uma coordenada em um plano cartesiano.
    """

    def __init__(self, x: float, y: float):
        """
        Cria uma coordenada.

        :param x: a posição do eixo x no plano cartesiano.
        :param y: a posição do eixo y no plano cartesiano.
        """
        self.x = x
        self.y = y

    def distance_to(self, other: 'Position') -> float:
        """
        Calcula a distância para outra coordenada.

        :param other: a outra coordenada.
        :return: a distância entre esta e a outra coordenada.
        """
        return distance((self.x, self.y), (other.x, other.y))

    def __eq__(self, other):
        return isinstance(other, Position) and self.x == other.x and self.y == other.y

    def __hash__(self):
        result = self.x
        result = 31 * result + self.y
        return result

    def __repr__(self):
        return f'Position({self.x}, {self.y})'
