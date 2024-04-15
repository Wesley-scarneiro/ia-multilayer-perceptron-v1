class Parameters:

    def __init__(self, 
                 learning_rate: float):
        self.__learning_rate = learning_rate
        self.__bias_value = 1
        self.__total_inputs = 120
    
    @property
    def learning_rate(self) -> float:
        return self.__learning_rate
    
    @property
    def bias_value(self) -> int:
        return self.__bias_value
    
    @property
    def total_inputs(self) -> int:
        return self.__total_inputs