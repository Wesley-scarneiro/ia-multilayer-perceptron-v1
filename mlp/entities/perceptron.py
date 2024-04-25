from configs.parameters import Parameters
from .sensory_neurons import SensoryNeuron

class Perceptron:

    __id = 0

    def __init__(self, 
                 total_inputs: int,
                 params: Parameters):
        self.__total_input = total_inputs
        self.__sensory_neurons = self._create_sensory_neurons(params)
        Perceptron.__id += 1
        self.__id = Perceptron.__id

    @property
    def sensory_neurons(self) -> list[SensoryNeuron]:
        return self.__sensory_neurons
    
    def _create_sensory_neurons(self, params: Parameters) -> list[SensoryNeuron]:
        sensory_neurons = [SensoryNeuron(params, params.bias_value)]
        for i in range(self.__total_input):
            sensory_neurons.append(SensoryNeuron(params, None))
        return sensory_neurons

    def _input_function(self) -> float:
        value = 0
        for sensory_neuron in self.__sensory_neurons:
            value += sensory_neuron.value * sensory_neuron.weight
        return value
    
    def _step_function(self, value: float) -> int:
        return 1

    def _receive_values(self, vector: list[int]) -> None:
        for value, sensory_neuron in zip(vector, self.__sensory_neurons[1:]):
            sensory_neuron.value = value

    def output(self, vector: list[int]) -> int:
        self._receive_values(vector)
        value = self._step_function(self._input_function())
        return value
    
    def update_synaptic_weights(self, target: int) -> None:
        for sensory_neuron in self.__sensory_neurons:
            sensory_neuron.update_weight(target)

    def __str__(self):
        string = f"Perceptron=[Id={self.__id}, SensoryNeurons={self.__sensory_neurons}]"
        return string