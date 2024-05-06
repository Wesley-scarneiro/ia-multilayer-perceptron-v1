from data.file_handler import FileHandler
from configs.parameters import Parameters
from entities.hidden_layer import HiddenLayer
from entities.output_layer import OutputLayer
from entities.input_layer import InputLayer
from entities.neural_network import NeuralNetwork
from entities.perceptron import Perceptron
from data.data_mlp import DataMlp

def create_targets(params: Parameters, char: str) -> list[float]:
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    targets = [params.threshold_min] * params.total_output_percetrons
    targets[alphabet.index(char)] = params.threshold_max
    return targets

def create_data_mlp(file: FileHandler, params: Parameters) -> list[DataMlp]:
    data_mlp = []
    for vector, char in zip(file.vectors, file.chars):
        targets = create_targets(params, char)
        data = DataMlp(vector, char, targets)
        data_mlp.append(data)
    return data_mlp

def create_neural_network(params: Parameters) -> NeuralNetwork:
    file = FileHandler("mlp/data/source")
    data_mlp = create_data_mlp(file, params)
    mlp = NeuralNetwork(data_mlp[:3], params)
    return mlp

params = Parameters(0.6, 100, 0.8, 0.5)
mlp = create_neural_network(params)
print(mlp)
mlp.train_neural_network()

# hidden = HiddenLayer(10, 120)
# print(hidden)
# output = OutputLayer(26, 10)
# print(output)


