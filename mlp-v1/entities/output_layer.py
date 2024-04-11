from .neural_layer import NeuralLayer

class OutputLayer(NeuralLayer):

    def __init__(self, total_collection: int, total_perceptrons: int):
        super().__init__(total_collection, total_perceptrons)