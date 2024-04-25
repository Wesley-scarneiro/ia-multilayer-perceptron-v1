from data.file_handler import FileHandler

class TestFileHandler:

    def test_fileHandler_read_files_vectors(self):
        fileHandler = file = FileHandler("mlp/data/source")
        assert len(fileHandler.vectors) == 1326
    
    def test_fileHandler_read_files_chars(self):
        fileHandler = file = FileHandler("mlp/data/source")
        assert len(fileHandler.chars) == 1326