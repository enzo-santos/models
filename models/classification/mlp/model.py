from typing import List, Optional, Union, Tuple

from models.classification.mlp.gradient import Gradient
from models.classification.mlp.perceptron import UpdatingPerceptron, Perceptron, InputPerceptron
from models.utils.linear_alg import Vector, Matrix
from models.utils.lists import cols

Layer = List[Perceptron]
UpdatingLayer = List[UpdatingPerceptron]


class MLPClassifier:
    """
    Representa uma rede neural com múltiplos neurônios em um contexto de classificação binária.

    ####
    Referências
    ####

    - https://codesachin.wordpress.com/2015/12/06/backpropagation-for-dummies/

    - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    """

    def __init__(self, *, n_layers: int, layer_size: int, learning_rate: float = 0.3, n_generations: int = 100):
        """
        Constrói uma rede neural.

        :param n_layers: número de camadas ocultas dessa rede.
        :param layer_size: número de neurônios em cada camada oculta dessa rede.
        :param learning_rate: taxa de aprendizado dos neurônios dessa rede.
        :param n_generations: número de gerações dessa rede.
        """

        self.n_layers: int = n_layers
        """
        O número de camadas ocultas dessa rede neural.
        
        Essa rede neural terá ``n_layers`` + 2 camadas (camada de entrada e de saída).
        """

        self.layer_size: int = layer_size
        """
        O número de neurônios presentes em cada camada oculta dessa rede neural.
        """

        self.n_generations: int = n_generations
        """
        O número de gerações dessa rede neural.
        """

        self.learning_rate: float = learning_rate
        """
        A taxa de aprendizagem dos neurônios dessa rede neural.
        """

        self.layers: Optional[List[Layer]] = None
        """
        As camadas dessa rede neural.
        
        Caso seja `None`, o método ``fit`` ainda não foi chamado.
        """

        self._class_map: Optional[Tuple[float, float]] = None
        """
        Mapeamento de classes dessa rede neural.
        
        Cada valor dessa tupla é uma classe fornecida pelo usuário (por ser um algoritmo de classificação binária, 
        a tupla possui apenas dois elementos). O valor da posição 0 corresponde ao valor da classe negativa e o valor
        da posição 1 ao valor da classe positiva). Como a função sigmoide apenas gera valores entre 0 e 1, esse 
        mapeamento é utilizado para normalizar os dados de entrada para esses dois valores.
        
        Caso seja `None`, o método ``fit`` ainda não foi chamado.
        """

    # noinspection NonAsciiCharacters,PyPep8Naming
    def _update_hidden_weights(self, previous_layer: UpdatingLayer, current_layer: UpdatingLayer,
                               next_layer: UpdatingLayer):
        """
        Atualiza os pesos de camada oculta da rede neural.

        Como é uma camada oculta, o cálculo dos novos pesos também depende dos pesos já calculados na camada posterior.

        :param previous_layer: a camada anterior (isto é, que age como entrada) à camada a ser atualizada.
        :param current_layer: a camada a ser atualizada.
        :param next_layer: a camada posterior (isto é, que age como saída) à camada a ser atualizada.
        """
        for i, current_perceptron in enumerate(current_layer):
            Δws: Vector = []
            a_i: float = current_perceptron.last_output

            for previous_perceptron in previous_layer:
                a_j: float = previous_perceptron.last_output

                δe: Gradient = Gradient()
                δe.δa = 0.0
                for next_perceptron in next_layer:
                    a_k: float = next_perceptron.last_output
                    w_k: float = next_perceptron.weights[i]
                    δe_k: Gradient = next_perceptron.gradients[i]

                    δe.δa += w_k * a_k * (1 - a_k) * δe_k.δa

                δe.δw = a_j * a_i * (1.0 - a_i) * δe.δa
                current_perceptron.put(δe)

                Δws.append(-self.learning_rate * δe.δw)

            weights: Vector = current_perceptron.weights
            assert len(weights) > 0
            assert len(weights) == len(current_perceptron.gradients)
            current_perceptron.update([weight + Δw for weight, Δw in zip(weights, Δws)])

    # noinspection NonAsciiCharacters,PyPep8Naming
    def _update_output_weights(self, previous_layer: UpdatingLayer, current_layer: UpdatingLayer, y_true: float):
        """
        Atualiza os pesos de camada de saída da rede neural.

        Como é uma camada de saída, o cálculo dos novos pesos apenas depende da saída esperada do algoritmo.

        :param previous_layer: a camada anterior (isto é, que age como entrada) à camada a ser atualizada.
        :param current_layer: a camada a ser atualizada.
        :param y_true: a saída esperada para a última saída produzida por essa camada.
        """
        for current_perceptron in current_layer:
            Δws: Vector = []
            a_i: float = current_perceptron.last_output

            for previous_perceptron in previous_layer:
                a_j: float = previous_perceptron.last_output
                t_i: float = y_true

                δe: Gradient = Gradient()
                δe.δa = - (t_i - a_i)
                δe.δw = a_j * a_i * (1.0 - a_i) * δe.δa
                current_perceptron.put(δe)

                Δws.append(-self.learning_rate * δe.δw)

            weights: Vector = current_perceptron.weights
            assert len(weights) > 0
            assert len(weights) == len(current_perceptron.gradients)
            current_perceptron.update([weight + Δw for weight, Δw in zip(weights, Δws)])

    def _update_weights(self, sample: Vector, y_true: float):
        """
        Atualiza os pesos das camadas da rede neural por backpropagation.

        :param sample: a amostra cujos novos pesos serão adaptados de acordo.
        :param y_true: a saída esperada para essa amostra.
        """
        self.transform(sample)

        layers: List[UpdatingLayer]
        layers = [[UpdatingPerceptron(perceptron) for perceptron in layer] for layer in self.layers]

        for i_l, current_layer in reversed(list(enumerate(layers[1:], start=1))):
            previous_layer: UpdatingLayer = layers[i_l - 1]

            if i_l < len(layers) - 1:
                next_layer: UpdatingLayer = layers[i_l + 1]
                self._update_hidden_weights(previous_layer, current_layer, next_layer)
            else:
                self._update_output_weights(previous_layer, current_layer, y_true)

    def fit(self, X: Matrix, y: Vector) -> 'MLPClassifier':
        """
        Ajusta o classificador ao conjunto de dados de treinamento.

        O modo de atualização dos pesos é `online`, ocorrendo a cada chamada desse método.

        :param X: as características das amostras de treinamento.
        :param y: as classes das amostras de treinamento. Deve possuir apenas valores 0 e 1.
        :return: o classificador ajustado.
        """

        classes = set(y)
        if len(classes) > 2:
            raise ValueError("more than two classes found in y")

        self._class_map = tuple(sorted(classes))

        n_cols: int = cols(X)

        # Inicializa as camadas
        bias: float
        previous_layer_size: int

        layers: List[Layer] = [[InputPerceptron() for _ in range(n_cols)]]
        previous_layer_size = n_cols

        if self.n_layers > 0:
            bias = 0.0
            layers.append([Perceptron(input_size=previous_layer_size, bias=bias) for _ in range(self.layer_size)])
            previous_layer_size = self.layer_size

            for _ in range(self.n_layers - 1):
                bias = 0.0
                layers.append([Perceptron(input_size=previous_layer_size, bias=bias) for _ in range(self.layer_size)])
                previous_layer_size = self.layer_size

        bias = 0.0
        layers.append([Perceptron(input_size=previous_layer_size, bias=bias)])
        self.layers = layers

        # Atualiza os pesos por meio de backpropagation
        for _ in range(self.n_generations):
            for sample, outcome in zip(X, y):
                self._update_weights(sample, self._class_map.index(outcome))

        return self

    def transform(self, sample: Vector) -> Vector:
        """
        Transforma uma amostra em um conjunto de valores por meio dessa rede neural.

        :param sample: a amostra a ser transformada.
        :return: o conjunto de valores produzidos pela camada de saída após o algoritmo de rede neural ser aplicado. O
                    número de elementos nesse conjunto é igual ao número de neurônios na camada de saída dessa rede.
        """
        input_layer, *hidden_layers, output_layer = self.layers

        output: Vector = []
        for perceptron, feature in zip(input_layer, sample):
            output.append(perceptron.predict([feature]))

        for layer in hidden_layers:
            output = [perceptron.predict(output) for perceptron in layer]

        return [perceptron.predict(output) for perceptron in output_layer]

    def predict(self, X: Union[Vector, Matrix]) -> Union[float, Vector]:
        """
        Prediz as classes dos dados fornecidos.

        :param X: as características das amostras a serem previstas.
        :return: a(s) classe(s) das amostras fornecidas.
        """
        class_map = self._class_map
        if class_map is None:
            raise ValueError("you must call 'fit' before calling 'predict'")

        is_vector = all(isinstance(v, (float, int)) for v in X)
        X = [X] if is_vector else X

        outcomes: Vector = []
        for feature in X:
            values = self.transform(feature)
            assert len(values) == 1
            value = values[0]

            outcome = min((0, 1), key=lambda x: abs(x - value))
            if is_vector:
                return class_map[outcome]

            outcomes.append(class_map[outcome])

        return outcomes
