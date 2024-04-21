
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self, maze, **kwargs):
        self.environment = maze
        self.name = kwargs.get("name", "model")

    def load_model(self, filename): # load a previously trained model
        pass

    def save_model(self, filename): # save a trained model
        pass 

    def train_model(self, stop_at_convergence=False, **kwargs): # train the model
        pass

    @abstractmethod
    def q_values(self, state): # return the Q values for the state
        pass

    @abstractmethod
    def predict_value(self, state): # return the action with the highest Q value
        pass
