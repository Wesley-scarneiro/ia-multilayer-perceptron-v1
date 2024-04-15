class CharData:

    def __init__(self,
                 values: list[int],
                 char: str):
        self.__values = values
        self.__char = char
    
    @property
    def values(self) -> list[int]:
        return self.__values
    
    @property
    def char(self) -> str:
        return self.__char
    
    def __repr__(self) -> str:
        return f"CharData=[values={self.__values}, char={self.__char}]"