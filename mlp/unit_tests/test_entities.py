from entities.input_layer import InputLayer
from data.file_handler import FileHandler
from entities.hidden_layer import HiddenLayer
from mlp.entities.parameters import Parameters
from entities.perceptron_collection import PerceptronCollection
from entities.output_layer import OutputLayer
from entities.neural_network import NeuralNetwork

class TestEntities:

    def test_input_layer_vectors(self):
        file = FileHandler("mlp/data/source")
        input_layer = InputLayer(file.vectors)
        assert input_layer.total_vectors == 1326
    
    def test_PerceptronCollection(self):
        params = Parameters(0.5)
        file = FileHandler("mlp/data/source")
        collection = PerceptronCollection(params, 10, params.vector_dimension)
        
        expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        output = collection.output(file.vectors[0])
        assert expected == output
    
    def test_hidden_layer_1_sublayer_output(self):
        params = Parameters(0.5)
        file = FileHandler("mlp/data/source")
        hidden_layer = HiddenLayer(params, 1, 10)

        expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        output = hidden_layer.output(file.vectors[0])
        assert output == expected
    
    # Com step function retornando 1
    def test_hidden_layer_2_sublayer_output(self):
        params = Parameters(0.5)
        file = FileHandler("mlp/data/source")
        hidden_layer = HiddenLayer(params, 2, 10)

        expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        output = hidden_layer.output(file.vectors[0])
        assert output == expected

    # Com step function retornando 1
    def test_output_layer_output(self):
        params = Parameters(0.5)
        file = FileHandler("mlp/data/source")
        output_layer = OutputLayer(params, 10, 120)

        expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        output = output_layer.output(file.vectors[0])
        assert output == expected

    def test_feedforward(self):
        params = Parameters(0.5)
        file = FileHandler("mlp/data/source")
        input_layer = InputLayer([file.vectors[0]])
        hidden_layer = HiddenLayer(params, 2, 10)
        output_layer = OutputLayer(params, 10, 10)
        neural_network = NeuralNetwork(input_layer, hidden_layer, output_layer)

        expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        output = neural_network.output()
        assert output == expected
        
