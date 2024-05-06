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
    
    @property
    def perceptrons(self) -> list[Perceptron]:
        return self.__perceptrons
    
    @property
    def vector_dimension(self) -> int:
        return self.__vector_dimension
    
    @abstractmethod
    def update_weights():
        pass