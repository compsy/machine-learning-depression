from sklearn.grid_search import GridSearchCV

from machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn import svm
import numpy as np


class SupportVectorClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names, model_type='classification')
        self.skmodel = svm.SVC(verbose=verbosity, kernel='poly', degree=2, C=600000)

    def variable_to_validate(self):
        return 'degree'

    def predict_for_roc(self, x_data):
        return self.skmodel.decision_function(x_data)
