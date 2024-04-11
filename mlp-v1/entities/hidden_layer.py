from .neural_layer import NeuralLayer

class HiddenLayer(NeuralLayer):

    def __init__(self, total_collection: int, total_percetrons: int):
        super().__init__(total_collection, total_percetrons)