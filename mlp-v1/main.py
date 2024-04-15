from data.file_handler import FileHandler

file = FileHandler("mlp-v1/data/source")
charData = file.get_char_data()
print(len(charData))
print(charData[913])
