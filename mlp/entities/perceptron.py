import math
from .sensory_neurons import SensoryNeuron

class Perceptron:

    __id = 0

    def __init__(self, vector_dimension: int):
        self.__sensory_neurons = self._create_sensory_neurons(vector_dimension)
        Perceptron.__id += 1
        self.__id = Perceptron.__id
        self. __input_function_cache = None

    @property
    def weights(self) -> list[float]:
        return [sn.weight for sn in self.__sensory_neurons]
    
    def _create_sensory_neurons(self, vector_dimension: int) -> list[SensoryNeuron]:
        sensory_neurons = [SensoryNeuron(1)]
        for i in range(vector_dimension):
            sensory_neurons.append(SensoryNeuron(None))
        return sensory_neurons

    def __input_function(self) -> float:
        if self.__input_function_cache is None:
            value = 0
            for sensory_neuron in self.__sensory_neurons:
                value += sensory_neuron.value * sensory_neuron.weight
            self.__input_function_cache = value
        return self.__input_function_cache
    
    def __sigmoid_function(self, value: float) -> float:
        result =  1 / (1 + math.exp(-value))
        return max(0.00000000001, min(result, 0.9999999999))

    ## quando usada a derivada não ajusta
    def __derived_sigmoid_function(self, value: float) -> float:
        sigmoid_value = self.__sigmoid_function(value)
        result = sigmoid_value * (1 - sigmoid_value)
        return result

    def _receive_values(self, vector: list[int]) -> None:
        self.__input_function_cache = None
        for value, sensory_neuron in zip(vector, self.__sensory_neurons[1:]):
            sensory_neuron.value = value

    def output(self, vector: list[int]) -> float:
        self._receive_values(vector)
        value_input_function = self.__input_function()
        value_step_function = self.__sigmoid_function(value_input_function)
        return value_step_function
    
    def __output_error_information_term(self, target: float, output: float) -> float:
        error_rate = (target - output) * self.__derived_sigmoid_function(self.__input_function())
        return error_rate
    
    def weighted_output_error_information_term(self, target: float, output: float) -> list[float]:
        error_information_term = self.__output_error_information_term(target, output)
        error_informations = []
        for sensory_neuron in self.__sensory_neurons:
            error = error_information_term * sensory_neuron.weight
            error_informations.append(error)
        return error_informations
    
    def __hidden_error_information_term(self, output_layer_error: float) -> float:
        error_rate = self.__derived_sigmoid_function(self.__input_function()) * output_layer_error
        return error_rate

    def update_weights_ouput_layer(self, learning_rate: float, target: float, output: float) -> None:
        for sensory_neuron in self.__sensory_neurons:
            error_information_term = self.__output_error_information_term(target, output)
            sensory_neuron.update_weight(learning_rate, error_information_term)
    
    def update_weights_hidden_layer(self, learning_rate: float, output_layer_error: float) -> None:
        for sensory_neuron in self.__sensory_neurons:
            error_information_term = self.__hidden_error_information_term(output_layer_error)
            sensory_neuron.update_weight(learning_rate, error_information_term)

    def __str__(self):
        string = f"Perceptron=[Id={self.__id}, VectorDimension= {self.__vector_dimension}, SensoryNeurons={self.__sensory_neurons}]"
        return string