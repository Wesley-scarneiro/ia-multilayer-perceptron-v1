from .perceptron import Perceptron

class PerceptronCollection:

    def __init__(self, total_perceptrons: int):
        self.__ = self.__create_perceptrons(total_perceptrons)

    def __create_perceptrons(self, total_perceptrons: int) -> list[Perceptron]:
        perceptrons = []
        for i in range(total_perceptrons):
            perceptrons.append(Perceptron())
        return perceptrons