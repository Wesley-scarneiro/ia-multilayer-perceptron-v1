from entities.hidden_layer import HiddenLayer
from entities.output_layer import OutputLayer
from data.data_mlp import DataMlp
from configs.parameters import Parameters
from .input_layer import InputLayer
import math

'''
    Representa um multilayer perceptron (MLP)
'''
class NeuralNetwork:
    
    __id = 0

    def __init__(self,
                 data_mlp: list[DataMlp],
                 params: Parameters):
        self.__params = params
        self.__input_layer = InputLayer(data_mlp)
        self.__hidden_layer = HiddenLayer(params.total_hidden_perceptrons, params.input_vector_dimension)
        self.__output_layer = OutputLayer(params.total_output_percetrons, params.total_hidden_perceptrons)
        self.__error_info = []
        self.__final_era = None
        self.__final_error = 0
        NeuralNetwork.__id += 1
        self.__id = NeuralNetwork.__id
    
    @property
    def hidden_layer(self) -> HiddenLayer:
        return self.__hidden_layer
    
    @property
    def output_layer(self) -> OutputLayer:
        return self.__output_layer

    @property
    def error_info(self) -> list[(int, float)]:
        return self.__error_info

    # Processa um vetor de dados pelas camadas da MLP
    def __feedforward(self, vector: list[int]) -> list[float]:
        hidden_layer_output = self.__hidden_layer.output(vector)
        output = self.__output_layer.output(hidden_layer_output)
        return output
    
    # Realiza uma iteração completa dos dados e acumula o erro do aprendizado por era
    def __iterate_data(self) -> float:
        accumulated_error = 0
        for data in self.__input_layer.data:
            outputs = self.__feedforward(data.vector)
            accumulated_error += self.__mean_squared_error(outputs, data.target)
            output_error_rate = self.__output_layer.update_weights(self.__params.learning_rate, data.target, outputs)
            self.__hidden_layer.update_weights(self.__params.learning_rate, output_error_rate)
        return accumulated_error

    # Calcula o erro médio quadrático da rede
    def __mean_squared_error(self, outputs: list[float], targets: list[int]) -> float:
        error = 0
        for output, target in zip(outputs, targets):
            error += math.pow((target - output), 2)
        error /= 2
        return error
    
    # Realiza o treinamento da MLP
    def train_neural_network(self) -> None:
        current_error = 0
        print("Start learn")
        for era in range(self.__params.total_eras):
            previous_error = current_error
            current_error = self.__iterate_data()
            self.__error_info.append((era, current_error))
            print(f"- Error[{era}] = {current_error}")
            if current_error < self.__params.error_rate:
                break
        self.__final_era = era
        self.__final_error = current_error
        print("fineshed")
    
    # Gera a saída da rede para uma data entrada
    def output(self, vector: list[int]) -> list[float]:
        return self.__feedforward(vector)
    
    def __repr__(self) -> str:
        return f'''MLP=
    [
        - Id={self.__id},
        - {self.__params},
        - {self.__input_layer},
        - {self.__hidden_layer},
        - {self.__output_layer}
        - Final_era= {self.__final_era}
        - Final_error = {self.__final_error}
    ]'''
