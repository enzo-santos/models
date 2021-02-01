from typing import Optional


# noinspection NonAsciiCharacters
class Gradient:
    """
    Representa um gradiente de uma função erro.
    """

    __slots__ = ('δa', 'δw')

    def __init__(self, δa: Optional[float] = None, δw: Optional[float] = None):
        self.δa: Optional[float] = δa
        """
        A derivada parcial do erro em relação à ativação (saída) de um perceptron.
        
        Em outras palavras, esse valor indica o quanto uma alteração na saída de um perceptron afeta o seu erro.
        """

        self.δw: Optional[float] = δw
        """
        A derivada parcial do erro em relação ao peso de um perceptron.
        
        Em outras palavras, esse valor indica o quanto uma alteração em um peso de um perceptron afeta o seu erro.
        """

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Gradient) and self.δa == o.δa and self.δw == o.δw
