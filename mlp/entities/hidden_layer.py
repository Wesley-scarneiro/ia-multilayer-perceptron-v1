'''
    Representa a camada oculta da MLP.
    Subclasse da NeuralLayer, que representa um abstração de uma
    camada neural e suas operações básicas (independente da camada oculta ou de saída).
'''

from entities.neural_layer import NeuralLayer

class HiddenLayer(NeuralLayer):

    def __init__(self, total_perceptrons: int, vector_dimension: int):
        super().__init__(total_perceptrons, vector_dimension)

    # Realiza a atualização dos pesos de cada perceptron da camada
    def update_weights(self, learning_rate: float, output_layer_errors: list[list[float]], inputs: list[float]) -> None:
        for perceptron in self.perceptrons:
            index = self.perceptrons.index(perceptron)
            output_layer_error = self.__output_layer_error(output_layer_errors, index)
            perceptron.update_weights_hidden_layer(learning_rate, output_layer_error, inputs)

    # Calcula a contribuição do erro de um perceptron da camada oculta para o ajuste do peso
    def __output_layer_error(self, output_layer_errors: list[list[float]], index: int):
        value = 0
        for errors in output_layer_errors:
            value += errors[index]
        return value

    def __repr__(self) -> str:
        return f'''HiddenLayer=
        [
            - TotalPerceptrons={len(self.perceptrons)}, 
            - VectorDimension={self.vector_dimension}
        ]'''
        