from .perceptron import Perceptron
from configs.parameters import Parameters

class PerceptronCollection:

    def __init__(self,
                 params: Parameters,
                 total_perceptrons: int,
                 total_inputs: int):
        self.__perceptrons = self.__create_perceptrons(params, total_perceptrons, total_inputs)

    def __create_perceptrons(self, 
                             params: Parameters, 
                             total_perceptrons: int,
                             total_inputs: int) -> list[Perceptron]:
        perceptrons = []
        for i in range(total_perceptrons):
            perceptrons.append(Perceptron(total_inputs, params))
        return perceptrons
    
    def output(self, vector: list[int]) -> list[int]:
        output = []
        for perceptron in self.__perceptrons:
            output.append(perceptron.output(vector))
        return output

    def update_synaptic_weights(self, targets: list[int]) -> None:
        pass