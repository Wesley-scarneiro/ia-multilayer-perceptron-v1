'''
    Uma abstração de uma camada em uma rede neural.
    Esta abstração visa simplificar a implementação da camada oculta
    e de saída de uma MLP, visto que, ambas possuem comportamentos em comum.
    Contém uma lista de perceptrons e dimensão dos seus vetores de entrada.

    A camada oculta e de saída devem implementar somente o método abstrato update_weights()
'''

from abc import ABC, abstractmethod
from entities.perceptron import Perceptron

class NeuralLayer(ABC):
    
    def __init__(self, total_perceptron: int, vector_dimension: int):
        self.__perceptrons = self.__create_perceptrons(total_perceptron, vector_dimension)
        self.__vector_dimension = vector_dimension

    def __create_perceptrons(self, total_perceptrons: int , vector_dimension: int) -> list[Perceptron]:
        perceptrons = []
        for i in range(total_perceptrons):
            perceptrons.append(Perceptron(vector_dimension))
        return perceptrons
    
    def output(self, vector: list[float]) -> list[float]:
        output = []
        for perceptron in self.__perceptrons:
            output.append(perceptron.output(vector))
        return output
    
    def carry_weights(self, weights_list: list[list[float]]):
        for perceptron, weights in zip(self.perceptrons, weights_list):
            perceptron.weights = weights
    
    @property
    def perceptrons(self) -> list[Perceptron]:
        return self.__perceptrons
    
    @property
    def vector_dimension(self) -> int:
        return self.__vector_dimension
    
    @property
    def weights(self) -> list[list[float]]:
        weights = []
        for perceptron in self.__perceptrons:
            weights.append(perceptron.weights)
        return weights
    
    @abstractmethod
    def update_weights():
        pass