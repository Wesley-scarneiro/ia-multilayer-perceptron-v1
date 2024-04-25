from data.char_data import CharData
from entities.hidden_layer import HiddenLayer

class InputLayer:

    def __init__(self, vectors: list[list[int]]):
        self.__vectors = vectors
    
    @property
    def vectors(self) -> list[list[int]]:
        self.__vectors
    
    def vector(self, vector_index: int) -> list[int]:
        return self.__vectors[vector_index]
    
    @property
    def total_vectors(self) -> int:
        return len(self.__vectors)
