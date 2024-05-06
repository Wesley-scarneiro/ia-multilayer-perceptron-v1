'''
    Representa os dados de treinamento da MLP
        - vector: dados de entrada
        - char: caractere que é representado pelo vetor
        - target: saída esperada da MLP para os dados de entrada
'''
class DataMlp:

    def __init__(self,
                 vector: list[int],
                 char: str=None,
                 target: list[float]=None):
        self.__vector = vector
        self.__char = char
        self.__target = target
    
    @property
    def vector(self) -> list[int]:
        return self.__vector
    
    @property
    def char(self) -> str:
        return self.__char
    
    @property
    def target(self) -> list[float]:
        return self.__target
    
    def __repr__(self) -> str:
        return f"DataMlp=[vector={self.__vector}, char={self.__char}, target={self.__target}]"