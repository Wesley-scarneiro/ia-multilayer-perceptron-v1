from .perceptron import Perceptron

class PerceptronCollection:

    def __init__(self, total_perceptrons: int, vector_dimension: int):
        self.__vector_dimension = vector_dimension
        self.__perceptrons = self.__create_perceptrons(total_perceptrons, vector_dimension)

    def __create_perceptrons(self, total_perceptrons: int , vector_dimension: int) -> list[Perceptron]:
        perceptrons = []
        for i in range(total_perceptrons):
            perceptrons.append(Perceptron(vector_dimension))
        return perceptrons
    
    def output(self, vector: list[int]) -> list[float]:
        output = []
        for perceptron in self.__perceptrons:
            output.append(perceptron.output(vector))
        return output

    def update_weights_output_layer(self, learning_rate: float, targets: list[float], outputs: list[float]) -> None:
        for perceptron, target, output in zip(self.__perceptrons, targets, outputs):
            perceptron.update_weights_ouput_layer(learning_rate, target, output)

    def __repr__(self) -> str:
        return f"PerceptronCollection=[TotalPerceptrons={len(self.__perceptrons)}, VectorDimension={self.__vector_dimension}]"