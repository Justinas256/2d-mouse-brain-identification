from models.base_model import BaseModel
from data_loader.base_data_loader import BaseDataLoader


class BaseTrain(object):
    def __init__(self, model: BaseModel, data_loader: BaseDataLoader):
        self.model = model
        self.data_loader = data_loader

    def train(self):
        raise NotImplementedError
