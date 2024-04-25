from entities.output_layer import OutputLayer
from configs.parameters import Parameters
from .perceptron_collection import PerceptronCollection

class HiddenLayer:

    def __init__(self,
                 params: Parameters, 
                 total_sub_layers: int, 
                 total_perceptrons: int):
        self.__sub_layers = self.__create_sub_layers(params, total_sub_layers, total_perceptrons)
    
    def __create_sub_layers(self,
                            params: Parameters,  
                            total_sub_layers: int, 
                            total_perceptrons: int) -> list[PerceptronCollection]:
        sub_layers = []
        total_input = params.total_inputs
        for i in range(total_sub_layers):
            sub_layers.append(PerceptronCollection(params, total_perceptrons, total_input))
            total_input = total_perceptrons
        return sub_layers  
    
    def output(self, vector: list[int]) -> list[int]:
        output = []
        for sub_layer in self.__sub_layers:
            output = sub_layer.output(vector)
            vector = output
        return output
        