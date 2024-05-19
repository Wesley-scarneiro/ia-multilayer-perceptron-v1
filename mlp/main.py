from data.file_handler import FileHandler
from configs.parameters import Parameters
from entities.neural_network import NeuralNetwork
from data.data_mlp import DataMlp
from entities.report import Report
from datetime import datetime

def create_targets(params: Parameters, char: str) -> list[float]:
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    targets = [params.threshold_min] * params.total_output_percetrons
    targets[alphabet.index(char)] = params.threshold_max
    return targets

def create_data_mlp(params: Parameters) -> list[DataMlp]:
    file = FileHandler("mlp/data/source")
    data_mlp = []
    for vector, char in zip(file.vectors, file.chars):
        targets = create_targets(params, char)
        data = DataMlp(vector, char, targets)
        data_mlp.append(data)
    return data_mlp

def main():
    params = Parameters(0.5, 1, 120, 100, 26, 600)
    data_mlp = create_data_mlp(params)
    alphabet = 1000
    mlp = NeuralNetwork(data_mlp[:alphabet], params)
    mlp.train_neural_network()
    report = Report(mlp, f"{datetime.now().strftime('%y%m%d%H%M%S')}", data_mlp[alphabet:])
    report.report()

if __name__ == "__main__":
    main()




