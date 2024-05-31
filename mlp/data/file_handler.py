import os

'''
    Responsável por manipular os arquivos que contém os dados de treinamento
    (vetores, caracteres e targets) utlizados no MLP.
'''
class FileHandler:
    
    def __init__(self, dir_path: str):
        self.__vectors = self.__read_file_vectors(dir_path)
        self.__chars = self.__read_file_chars(dir_path)

    @property
    def vectors(self) -> list[list[int]]:
        return self.__vectors
    
    @property
    def chars(self) -> list[str]:
        return self.__chars

    def __line_format_vector(self, content: str) -> list[int]:
        content_list_str = content.replace(" ", "").replace("\n", "").split(",")
        content_list_int = []
        for i in content_list_str:
            if i != '':
                content_list_int.append(int(i))
        return content_list_int

    def __read_file_vectors(self, dir_path: str) -> list[list[int]]:
        file_path = os.path.join(dir_path, "vectors.txt")
        vectors = []
        with open(file_path, "r") as file:
            for line in file:
                vectors.append(self.__line_format_vector(line))
        return vectors

    def __read_file_chars(self, dir_path: str) -> list[str]:
        file_path = os.path.join(dir_path, "chars.txt")
        chars = []
        with open(file_path, "r") as file:
            for line in file:
                chars.append(line.replace("\n", ""))
        return chars