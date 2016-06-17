from machineLearningModels.machine_learning_model import MachineLearningModel
import numpy as np
from sklearn import ensemble
from sklearn import datasets


class BoostingModel(MachineLearningModel):

    def train(self):
        print(self.__class__.__name__ + ' >> Implement actual training model')
        return None
