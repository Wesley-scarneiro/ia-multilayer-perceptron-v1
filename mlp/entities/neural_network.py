from entities.hidden_layer import HiddenLayer
from entities.output_layer import OutputLayer
from data.data_mlp import DataMlp
from configs.parameters import Parameters
from .input_layer import InputLayer

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
        NeuralNetwork.__id += 1
        self.__id = NeuralNetwork.__id

    # Processa um vetor de dados pela mlp
    def __feedforward(self, vector: list[int]) -> list[float]:
        hidden_layer_output = self.__hidden_layer.output(vector)
        output = self.__output_layer.output(hidden_layer_output)
        return output
    
    # algoritmo incompleto
    def train_neural_network(self):
        data = self.__input_layer.data[0]
        print(f"- Data: {data.vector}")
        print(f"- Char: {data.char}")
        for i in range(100):
            outputs = self.__feedforward(data.vector)
            print(f"- Output[{i}]: {outputs}")
            if not self.__compare_outputs_targets(outputs, data.target):
                # Atualiza pesos da camada de saída e retorna a taxa de erro líquida
                output_error_rate = self.__output_layer.update_weights(self.__params.learning_rate, data.target, outputs)
                # Atualiza pesos da camada oculta
                self.__hidden_layer.update_weights(self.__params.learning_rate, output_error_rate)
    
    def __compare_outputs_targets(self, outputs: list[float], targets: list[float]) -> bool:
        for output, target in zip(outputs, targets):
            if target == self.__params.threshold_max:
                if output < target:
                    return False
            else:
                if output > target:
                    return False
        return True
            
    def __repr__(self) -> str:
        return f"MLP=[Id={self.__id}, {self.__params}, {self.__input_layer}, {self.__hidden_layer}, {self.__output_layer}]"
