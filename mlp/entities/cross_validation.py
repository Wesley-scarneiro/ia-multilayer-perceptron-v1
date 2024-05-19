from entities.neural_network import NeuralNetwork
from entities.report import Report
from data.data_mlp import DataMlp


class CrossValidation:

    def __init__(self, mlp: NeuralNetwork, report: Report, data: DataMlp):
        self.__mlp = mlp
        self.__report = report
        self.__data = data
    
    def __create_folds(self, data: DataMlp):
        folds = []
        max = len(DataMlp)/3
        for _