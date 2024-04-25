from data.file_handler import FileHandler
from configs.parameters import Parameters
from entities.hidden_layer import HiddenLayer
from entities.output_layer import OutputLayer
from entities.input_layer import InputLayer
from entities.neural_network import NeuralNetwork

file = FileHandler("mlp/data/source")
charData = file.vectors
print(len(charData))
print(charData[913])


params = Parameters(0.5)
file = FileHandler("mlp/data/source")
input_layer = InputLayer([file.vectors[0]])
hidden_layer = HiddenLayer(params, 2, 10)
output_layer = OutputLayer(params, 10, 10)
neural_network = NeuralNetwork(input_layer, hidden_layer, output_layer)

expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
output = neural_network.output()
assert output == expected