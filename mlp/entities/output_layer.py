from entities.neural_layer import NeuralLayer

class OutputLayer(NeuralLayer):

    def __init__(self, total_perceptrons: int, vector_dimension: int):
        super().__init__(total_perceptrons, vector_dimension)

    def update_weights(self, learning_rate: float, targets: list[float], outputs: list[float]) -> list[list[float]]:
        output_layer_errors = []
        for perceptron, target, output in zip(self.perceptrons, targets, outputs):
            output_layer_errors.append(perceptron.weighted_output_error_information_term(target, output))
            perceptron.update_weights_ouput_layer(learning_rate, target, output)
        return output_layer_errors
        
    def __repr__(self) -> str:
        return f"OutputLayer=[TotalPerceptrons={len(self.perceptrons)}, VectorDimension={self.vector_dimension}]"