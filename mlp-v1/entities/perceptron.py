from configs.parameters import Parameters
from .sensory_neurons import SensoryNeuron

class Perceptron:

    __id = 0

    def __init__(self, 
                 total_inputs: int,
                 params: Parameters):
        self.__sensory_neurons = self._create_sensory_neurons(total_inputs)
        self.__params = params
        Perceptron.__id += 1
        self.__id = Perceptron.__id

    @property
    def sensory_neurons(self) -> list[SensoryNeuron]:
        return self.__sensory_neurons
    
    def _create_sensory_neurons(self, inputs_total:int) -> list[SensoryNeuron]:
        sensory_neurons = [SensoryNeuron(Parameters.bias_value)]
        for i in range(inputs_total):
            sensory_neurons.append(SensoryNeuron(None))
        return sensory_neurons

    def _input_function(self) -> float:
        value = 0
        for sensory_neuron in self.__sensory_neurons:
            value += sensory_neuron.value * sensory_neuron.weight
        return value
    
    def _step_function(self, value: float) -> int:
        pass

    def _receive_values(self, list_values: list[int]) -> None:
        for value, sensory_neuron in zip(list_values, self.__sensory_neurons[1:]):
            sensory_neuron.value = value

    def output(self, list_values: list[int]) -> int:
        self._receive_values(list_values)
        value = self._step_function(self._input_function())
        return value
    
    def update_synaptic_weights(self, target: int) -> None:
        for sensory_neuron in self.__sensory_neurons:
            sensory_neuron.update_weight(target)

    def __str__(self):
        string = f"Perceptron=[Id={self.__id}, SensoryNeurons={self.__sensory_neurons}]"
        return string