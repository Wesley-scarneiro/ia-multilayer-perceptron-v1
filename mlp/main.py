from data.file_handler import FileHandler
from entities.parameters import Parameters
from data.data_mlp import DataMlp
from entities.cross_validation import CrossValidation
import json

def carry_weights(filename: str) -> list[list[list[float]]]:
    weight_list = []
    with open(filename, 'r') as file:
        for line in file:
            weight_list.append(json.loads(line))
    return weight_list

def create_targets(char: str) -> list[float]:
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    targets = [0] * 26
    targets[alphabet.index(char)] = 1
    return targets

def create_data_mlp() -> list[DataMlp]:
    file = FileHandler("mlp/data/source")
    data_mlp = []
    for vector, char in zip(file.vectors, file.chars):
        targets = create_targets(char)
        data = DataMlp(vector, char, targets)
        data_mlp.append(data)
    return data_mlp

def main():
    data_mlp = create_data_mlp()
    cross_validation = CrossValidation(data_mlp, 13)
    params = Parameters(0.5, 1, 120, 150, 26, 50)
    cross_validation.report(params)
    # params = Parameters(0.5, 1, 120, 150, 26, 50)
    # weight_list = carry_weights("mlp/logs/cross_13/cross_validation_3_240520132222-ok/mlp_weights_240520132222.log")
    # mlp = NeuralNetwork(params, weight_list=weight_list)
    # print(mlp.output(data_mlp[25].vector))

if __name__ == "__main__":
    main()




