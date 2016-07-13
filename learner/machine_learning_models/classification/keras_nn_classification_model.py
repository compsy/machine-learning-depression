from keras.optimizers import Adam

from machine_learning_models.machine_learning_model import MachineLearningModel
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
from pandas import DataFrame


class KerasNnClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names)
