class Parameters:

    def __init__(self, 
                 learning_rate: float,
                 total_hidden_perceptrons: int,
                 threshold_max: float,
                 threshold_min: float):
        self.__learning_rate = learning_rate
        self.__total_hidden_perceptrons = total_hidden_perceptrons
        self.__total_output_percetrons = 26
        self.__input_vector_dimension = 120
        self.__bias_value = 1
        self.__threshold_max = threshold_max
        self.__threshold_min = threshold_min
    
    @property
    def learning_rate(self) -> float:
        return self.__learning_rate
    
    @property
    def total_hidden_perceptrons(self) -> int:
        return self.__total_hidden_perceptrons
    
    @property
    def total_output_percetrons(self) -> int:
        return self.__total_output_percetrons
    
    @property
    def bias_value(self) -> int:
        return self.__bias_value
    
    @property
    def input_vector_dimension(self) -> int:
        return self.__input_vector_dimension
    
    @property
    def threshold_max(self) -> float:
        return self.__threshold_max
    
    @property
    def threshold_min(self) -> float:
        return self.__threshold_min
    
    def __repr__(self) -> str:
        return f"Params=[LearningRate={self.__learning_rate}, TotalHiddenPerceptrons={self.__total_hidden_perceptrons}]"