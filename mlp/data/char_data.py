class CharData:

    def __init__(self,
                 vector: list[int],
                 char: str):
        self.__vector = vector
        self.__char = char
    
    @property
    def vector(self) -> list[int]:
        return self.__vector
    
    @property
    def char(self) -> str:
        return self.__char
    
    def __repr__(self) -> str:
        return f"CharData=[values={self.__vector}, char={self.__char}]"