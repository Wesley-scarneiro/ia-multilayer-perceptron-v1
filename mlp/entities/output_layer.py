from entities.perceptron_collection import PerceptronCollection
from configs.parameters import Parameters

class OutputLayer:

    def __init__(self, 
                 params: Parameters,
                 total_perceptrons: int,
                 total_inputs):
        self.__perceptrons = PerceptronCollection(params, total_perceptrons, total_inputs)
    
    def output(self, vector: list[int]) -> list[int]:
        output = self.__perceptrons.output(vector)
        return output