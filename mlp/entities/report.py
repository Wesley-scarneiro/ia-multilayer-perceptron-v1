from entities.neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import logging
from data.data_mlp import DataMlp
import matplotlib.pyplot as plt
import numpy
import os


class Report:
    def __init__(self, mlp: NeuralNetwork, time_id, data: list[DataMlp]):
        self.__id = time_id
        self.__mlp = mlp
        self.__data = data
        self.__log_dir = f"mlp/logs/report_{self.__id}"
        self.__report_logger = self.__config_logging("report_mlp")
        self.__weights_logger = self.__config_logging("mlp_weights")

    def __config_logging(self, filename: str):
        os.makedirs(self.__log_dir, exist_ok=True) 
        logger = logging.getLogger(filename)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        file_handler = logging.FileHandler(f'{self.__log_dir}/{filename}_{self.__id}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def __generate_error_graph(self) -> None:
        eras, errors = zip(*self.__mlp.error_info)
        plt.plot(eras, errors)                     
        plt.xlabel('Eras')                         
        plt.ylabel('Accumulated error')
        plt.title('Accumulated error by eras')
        plt.savefig(f"{self.__log_dir}/graph_{self.__id}")
    
    def __learning_assessment(self) -> None:
        self.__report_logger.info("-- Learning assessment --")
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        total_hits = 0
        char_hits = []
        char_erros = []
        for data in self.__data:
            output = self.__mlp.output(data.vector)
            index = output.index(numpy.amax(output))
            char = f"{data.char}({self.__data.index(data)})"
            if (data.char == alphabet[index]):
                total_hits += 1
                char_hits.append(char)
            else:
                char_erros.append(char)
        self.__report_logger.info(f"- Total_hits = {total_hits}")
        self.__report_logger.info(f"- Total_errors = {len(self.__data) - total_hits}")
        self.__report_logger.info(f"- Char_hits = {char_hits}")
        self.__report_logger.info(f"- Char_errors = {char_erros}")

    def report(self) -> None:
        self.__report_logger.info("-- MLP info --")
        self.__report_logger.info(self.__mlp)
        self.__learning_assessment()
        self.__weights_logger.info(self.__mlp.hidden_layer.weights)
        self.__weights_logger.info(self.__mlp.output_layer.weights)
        self.__generate_error_graph()