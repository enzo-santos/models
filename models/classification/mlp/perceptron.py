import math
import random
from typing import Optional, List

from models.classification.mlp.gradient import Gradient
from models.utils.linear_alg import Vector
from models.utils.math import sigmoid


# noinspection NonAsciiCharacters,PyPep8Naming
class Perceptron:
    """
    Representa um perceptron (ou neurônio) de uma rede neural.

    Esse perceptron possui como função de ativação a função sigmoide.
    """

    __slots__ = ('bias', 'weights', '_last_input', '_last_output')

    def __init__(self, input_size: int, bias: float, *, weights: Optional[Vector] = None):
        self.bias: float = bias
        """
        Representa um bias a ser adicionado aos pesos em cada predição desse perceptron.
        """

        self.weights: Vector
        """
        Armazena os pesos a serem considerados a cada entrada fornecida a esse perceptron.
        """
        if weights is None:
            # He initialization
            self.weights = [random.gauss(mu=0, sigma=math.sqrt(2.0 / input_size)) for _ in range(input_size)]
        else:
            if len(weights) != input_size:
                raise ValueError("number of weights must be the same as input size")

            self.weights = weights

        self._last_input: Optional[Vector] = None
        self._last_output: Optional[float] = None

    @property
    def last_input(self) -> Optional[Vector]:
        """
        Armazena a última entrada enviada a esse perceptron.

        Caso seja `None`, o método ``predict`` ainda não foi chamado.
        """
        return self._last_input

    @property
    def last_output(self) -> Optional[float]:
        """
        Armazena a última saída produzida por esse perceptron.

        Caso seja `None`, o método ``predict`` ainda não foi chamado.
        """
        return self._last_output

    def predict(self, sample: Vector) -> float:
        """
        Classifica uma amostra de acordo com uma função de ativação.

        :param sample: a amostra a ser prevista.
        :return: a classe dessa amostra, entre 0 e 1.
        """
        if len(sample) != len(self.weights):
            raise ValueError(f"sample size ({len(sample)}) must match given input size ({len(self.weights)})")

        y: float = 1.0 * self.bias
        y += sum(value * weight for value, weight in zip(sample, self.weights))
        y = self.transform(y)

        self._last_input = sample
        self._last_output = y
        return y

    def transform(self, value: float) -> float:
        """
        Representa a função de ativação desse perceptron.

        :param value: o valor a ser transformado.
        :return: o valor de ativação respectivo.
        """
        return sigmoid(value)

    def update(self, weights: Vector):
        """
        Atualiza os pesos desse perceptron.

        :param weights: os novos pesos.
        """
        if self.weights is not None and len(self.weights) != len(weights):
            raise ValueError("old and new weights must have the same size")

        self.weights = weights


class InputPerceptron(Perceptron):
    """
    Representa um perceptron de entrada de uma rede neural.

    As propriedades desse perceptron são 1) recebe apenas um valor de entrada e 2) seu bias é zero, seu peso é um e não
    possui função de ativação (isto é, não altera a entrada).
    """

    def __init__(self):
        super().__init__(input_size=1, bias=0.0, weights=[1.0])

    def transform(self, value: float) -> float:
        return value


class UpdatingPerceptron(Perceptron):
    """
    Representa um perceptron de uma rede neural cujos pesos estão sendo atualizados.
    """

    def __init__(self, perceptron: Perceptron):
        super().__init__(len(perceptron.weights), perceptron.bias, weights=perceptron.weights)
        self._perceptron: Perceptron = perceptron

        self._last_input = perceptron.last_input
        self._last_output = perceptron.last_output

        self.gradients: List[Gradient] = []
        """
        Os gradientes desse perceptron.
        
        Cada gradiente está associado com um peso. Caso o gradiente `δe` de um peso seja calculado, é possível calcular
        a variação de peso que esse gradiente causa no peso original por meio da fórmula `- ε * δe.δw`, onde `ε` é a
        taxa de aprendizagem do perceptron.
        """

    def put(self, gradient: Gradient):
        """
        Adiciona um gradiente a esse perceptron.

        Na `i`-ésima chamada desse método, o gradiente a ser adicionado estará associado com o peso `i` em ``weights``.

        :param gradient: o gradiente a ser adicionado.
        """
        self.gradients.append(gradient)

    def update(self, weights: Vector):
        self._perceptron.update(weights)
