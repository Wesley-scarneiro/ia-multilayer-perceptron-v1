'''
    Trabalho 1 da disciplina de Inteligência Artificial.
    Implementação de uma multilayer perceptron com uma camada de entrada, 
    uma camada oculta e uma de saída.
    MLP implementada com cross validation e parada antecipada.
'''

import numpy as np
from data.file_handler import FileHandler
from entities.parameters import Parameters
from data.data_mlp import DataMlp
from entities.cross_validation import CrossValidation
from entities.neural_network import NeuralNetwork
import json

# Realiza a leitura dos pesos da camada oculta e de saída contidos em um arquivo
def carry_weights(filename: str) -> list[list[list[float]]]:
    weight_list = []
    with open(filename, 'r') as file:
        for line in file:
            weight_list.append(json.loads(line))
    return weight_list

# Constroi uma lista de alvos que serão utilizados no treinamento da MLP,
# de acordo com cada dado de treinamento obtido
def create_targets(char: str) -> list[float]:
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    targets = [0] * 26
    targets[alphabet.index(char)] = 1
    return targets

# Cria os dados de treinamento e de execução da MLP, a partir da leitura dos arquivos de dados
def create_data_mlp() -> list[DataMlp]:
    file = FileHandler("mlp/data/source")
    data_mlp = []
    for vector, char in zip(file.vectors, file.chars):
        targets = create_targets(char)
        data = DataMlp(vector, char, targets)
        data_mlp.append(data)
    return data_mlp

# Executa a rede aplicando o cross validation sobre o conjunto de dados
def run_cross_validation(data: DataMlp, params: Parameters, total_folds: int):
    cross_validation = CrossValidation(data, total_folds)
    cross_validation.report(params)

# Cria uma MLP com um conjunto de pesos já pré-definido por um treinamento
def run_with_carry_weights(data: list[DataMlp], params: Parameters, file_path: str, index_data: int):
    weight_list = carry_weights(file_path)
    mlp = NeuralNetwork(params, weight_list=weight_list)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    _input = data[index_data]
    output = mlp.output(_input.vector)
    char_result = alphabet[output.index(np.amax(output))]
    print(f"- Input: {_input.char}\n- Output: {char_result}\n- Match? {_input.char == char_result}")

def main():
    data_mlp = create_data_mlp()
    params = Parameters(0.5, 1, 120, 150, 26, 50)
    # run_cross_validation(data_mlp, params, 13)
    file_path = "mlp/logs/cross_13/cross_validation_0_24-05-22-23-53-36/mlp_weights_24-05-22-23-53-36.log"
    run_with_carry_weights(data_mlp, params, file_path, 899)

if __name__ == "__main__":
    main()




