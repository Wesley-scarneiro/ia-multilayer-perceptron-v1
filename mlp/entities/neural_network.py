from entities.hidden_layer import HiddenLayer
from entities.output_layer import OutputLayer
from .input_layer import InputLayer

'''
    Representa uma rede neural.
    A rede neural se comunica somente com a interface da camada de entrada.
'''
class NeuralNetwork:
    
    def __init__(self,
                 input_layer: InputLayer,
                 hidden_layer: HiddenLayer,
                 output_layer: OutputLayer):
        self.__input_layer = input_layer
        self.__hidden_layer = hidden_layer
        self.__output_layer = output_layer

    def __feedforward(self, vector_index: int) -> list[int]:
        vector = self.__input_layer.vector(vector_index)
        hidden_layer_output = self.__hidden_layer.output(vector)
        output = self.__output_layer.output(hidden_layer_output)
        return output
    
    # Algoritmo de treinamento...ainda imcompleto
    # testando o fluxo camada entrada --> camada oculta --> camada saída --> resultado
    def output(self) -> list[int]:
        return self.__feedforward(0)