from data.data_mlp import DataMlp

'''
    Representa a camada de entrada da MLP.
'''
class InputLayer:

    def __init__(self, data: list[DataMlp]):
        self.__data = data
    
    @property
    def data(self) -> list[DataMlp]:
        return self.__data

    def __repr__(self) -> str:
        return f'''InputLayer=
        [
            - TotalData = {len(self.__data)}
        ]'''