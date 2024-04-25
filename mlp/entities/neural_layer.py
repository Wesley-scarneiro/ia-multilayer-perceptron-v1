from abc import ABC, abstractmethod
from .perceptron_collection import PerceptronCollection

class NeuralLayer(ABC):

    def __init__(self, 
                 total_sub_layers: int, 
                 total_perceptrons: int):
        self._sub_layers = self.__create_sub_layers(total_sub_layers, total_perceptrons)
    
    def __create_sub_layers(self, total_sub_layers: int, total_perceptrons: int) -> list[PerceptronCollection]:
        sub_layers = []
        for i in range(total_sub_layers):
            sub_layers.append(PerceptronCollection(total_perceptrons))
        return sub_layers