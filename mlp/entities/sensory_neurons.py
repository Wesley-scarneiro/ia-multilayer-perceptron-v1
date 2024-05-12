from random import uniform

class SensoryNeuron:

    __id = 0
    
    def __init__(self, value: int):
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

    def update_weight(self, learning_rate: float, error_rate: float) -> None:
        self.__weight += (learning_rate * error_rate * self.__value)
    
    def _weight_random(self) -> float:
        return uniform(0, 0.01)
    
    def __str__(self):
        string = f"[Id={self.__id}, Value={self.__value}, Weight={self.__weight}]"
        return string
    
    def __repr__(self):
        string = f"[Id={self.__id}, Value={self.__value}, Weight={self.__weight}]"
        return string