import math
import random
import numpy as np

class Perceptron:

    __id = 0

    def __init__(self, vector_dimension: int):
        self.__weights = self._create_weights(vector_dimension)
        Perceptron.__id += 1
        self.__id = Perceptron.__id
        self. __input_function_result = 0
    
    def _random_weight(self) -> float:
        return random.uniform(0, 0.01)
    
    def _create_weights(self, vector_dimension: int) -> np.ndarray:
        weights = [self._random_weight()]
        for _ in range(vector_dimension):
            weights.append(self._random_weight())
        return np.array(weights)
        
    @property
    def weights(self) -> list[float]:
        return self.__weights.tolist()
    
    @weights.setter 
    def weights(self, weights: list[float]) -> None:
        self.__weights = np.array(weights)
    
    def __input_function(self, vector) -> None:
            self.__input_function_result = self.__weights[0]
            self.__input_function_result += np.dot(vector, self.__weights[1:])
    
    def __sigmoid_function(self, value: float) -> float:
        result =  1 / (1 + math.exp(-value))
        return max(0.00000000001, min(result, 0.9999999999))

    ## quando usada a derivada não ajusta
    def __derived_sigmoid_function(self, value: float) -> float:
        sigmoid_value = self.__sigmoid_function(value)
        result = sigmoid_value * (1 - sigmoid_value)
        return result

    def output(self, vector: list[int]) -> float:
        self.__input_function(np.array(vector))
        value = self.__sigmoid_function(self.__input_function_result)
        return value
    
    def __output_error_information_term(self, target: float, output: float) -> float:
        error_rate = (target - output) * self.__derived_sigmoid_function(self.__input_function_result)
        return error_rate
    
    def weighted_output_error_information_term(self, target: float, output: float) -> list[float]:
        error_information_term = self.__output_error_information_term(target, output)
        error_informations = np.multiply(self.__weights, error_information_term)
        return error_informations.tolist()
    
    def __hidden_error_information_term(self, output_layer_error: float) -> float:
        error_rate = self.__derived_sigmoid_function(self.__input_function_result) * output_layer_error
        return error_rate
    
    def update_weights_ouput_layer(self, learning_rate: float, target: float, _inputs: list[float], output: float) -> None:
        error_rate = self.__output_error_information_term(target, output)
        self.__weights[0] += learning_rate * error_rate
        self.__weights[1:] += (np.array(_inputs) * (learning_rate * error_rate))
    
    def update_weights_hidden_layer(self, learning_rate: float, output_layer_error: float, _inputs: list) -> None:
        error_rate = self.__hidden_error_information_term(output_layer_error)
        self.__weights[0] += learning_rate * error_rate
        self.__weights[1:] += (np.array(_inputs) * (learning_rate * error_rate))

    def __str__(self):
        string = f"Perceptron=[Id={self.__id}, VectorDimension= {len(self.__sensory_neurons)}, SensoryNeurons={self.__sensory_neurons}]"
        return string