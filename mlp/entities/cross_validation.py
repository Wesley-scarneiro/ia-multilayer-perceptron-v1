'''
    Implementa o algoritmo que realiza o treinamento da rede 
    com a estratégia do Cross Validation.
    A rede é treinada com k-1 folds e validada com um fold escolhido.
'''

from datetime import datetime
from entities.neural_network import NeuralNetwork
from entities.report import Report
from data.data_mlp import DataMlp
from entities.parameters import Parameters

class CrossValidation:

    def __init__(self, data: DataMlp, divider: int):
        self.__folds = self.__create_folds(data, divider)
    
    # Divida o conjunto de dados em K folds
    def __create_folds(self, data: DataMlp, divider: int):
        folds = []
        max_items = len(data)/divider
        fold = []
        for d in data:
            fold.append(d)
            if (len(fold) == max_items):
                folds.append(fold.copy())
                fold.clear()
        return folds

    # Treina a rede com K-1 folds e valida com um fold escolhido em K iterações
    def report(self, params: Parameters):
        for index in range(len(self.__folds)):
            temp_folds = self.__folds.copy()
            validation_fold = temp_folds[index]
            temp_folds.remove(validation_fold)
            traning_fold = sum(temp_folds, [])
            mlp = NeuralNetwork(params, traning_fold)
            mlp.train_neural_network()
            report = Report(mlp, f"{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}", validation_fold, dir_name=f"cross_validation_{index}")
            report.report()