from configs.parameters import Parameters
from random import uniform

class SensoryNeuron:

    __id = 0
    
    def __init__(self,
                params: Parameters, 
                value: int):
        self.__params = params
        self.__value = value
        self.__weight = self._weight_random()
        SensoryNeuron.__id += 1
        self.__id = SensoryNeuron.__id

    @property
    def value(self) -> int:
        return self.__value
    
    @value.setter
    def value(self, value: int) -> None:
        self.__value = value
    
    @property
    def weight(self) -> float:
        return self.__weight
    
    def update_weight(self, target: int) -> None:
        value = self.__weight + (self.__params.learning_rate * target * self.__value)
        self.__weight = round(value, 3)
    
    def _weight_random(self) -> float:
        return round(uniform(0, 1), 3)
    
    def __str__(self):
        string = f"[Id={self.__id}, Value={self.__value}, Weight={self.__weight}]"
        return string
    
    def __repr__(self):
        string = f"[Id={self.__id}, Value={self.__value}, Weight={self.__weight}]"
        return string