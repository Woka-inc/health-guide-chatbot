import json
from .base_data_loader import DataLoader

class JsonLoader(DataLoader):
    def __init__(self):
        super().__init__()

    def load(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data