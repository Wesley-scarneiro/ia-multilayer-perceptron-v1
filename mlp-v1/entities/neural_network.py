from .input_layer import InputLayer
from .hidden_layer import HiddenLayer
from .output_layer import OutputLayer

'''
    Representa uma rede neural com uma camada de entrada,
    uma camada oculta e uma camada de saída.
    As camadas oculta e de saída podem conter uma ou mais subcamadas.
'''
class NeuralNetwork:
    
    def __init__(self, 
                 input_layer: InputLayer, 
                 hidden_layer: HiddenLayer,
                 output_layer: OutputLayer):
        self.__input_layer = input_layer
        self.__hidden_layer = hidden_layer
        self.__output_layer = output_layer