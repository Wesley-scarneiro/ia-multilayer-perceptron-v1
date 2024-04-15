from abc import ABC, abstractmethod
from .perceptron_collection import PerceptronCollection

class NeuralLayer(ABC):

    def __init__(self, 
                 total_collection: int, 
                 total_perceptrons: int):
        self.__collection = self.__create_collection(total_collection, total_perceptrons)
    
    def __create_collection(self, total_collection: int, total_perceptrons: int) -> list[PerceptronCollection]:
        collection = []
        for i in range(total_collection):
            collection.append(PerceptronCollection(total_perceptrons))
        return collection