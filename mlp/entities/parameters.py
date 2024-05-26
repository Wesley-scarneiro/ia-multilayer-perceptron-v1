'''
    Representa os parâmetros de configuração da MLP.
'''

class Parameters:

    def __init__(self, 
                 learning_rate: float,
                 error_rate: float,
                 input_vector_dimension: int,
                 total_hidden_perceptrons: int,
                 total_output_percetrons: int,
                 total_eras: int):
        self.__learning_rate = learning_rate
        self.__error_rate = error_rate
        self.__input_vector_dimension = input_vector_dimension
        self.__total_hidden_perceptrons = total_hidden_perceptrons
        self.__total_output_percetrons = total_output_percetrons
        self.__total_eras = total_eras
        self.__bias_value = 1
        self.__threshold_max = 1
        self.__threshold_min = 0
    
    @property
    def learning_rate(self) -> float:
        return self.__learning_rate
    
    @property
    def error_rate(self) -> float:
        return self.__error_rate
    
    @property
    def total_hidden_perceptrons(self) -> int:
        return self.__total_hidden_perceptrons
    
    @property
    def total_output_percetrons(self) -> int:
        return self.__total_output_percetrons
    
    @property
    def total_eras(self) -> int:
        return self.__total_eras
    
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
        return f'''Params=
        [
            - learning_rate = {self.__learning_rate}, 
            - error_rate: float = {self.__error_rate}, 
            - input_vector_dimension = {self.__input_vector_dimension}, 
            - total_hidden_perceptrons = {self.__total_hidden_perceptrons}, 
            - total_output_percetrons = {self.__total_output_percetrons}, 
            - total_eras = {self.__total_eras}, 
            - bias_value = {self.__bias_value}, 
            - threshold_max = {self.__threshold_max}, 
            - threshold_min = {self.__threshold_min}
        ]'''