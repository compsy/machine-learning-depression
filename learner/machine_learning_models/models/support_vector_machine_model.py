from sklearn.grid_search import GridSearchCV

from machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn import svm
from numpy import logspace
import numpy as np


class SupportVectorRegressionModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names, model_type='regression')
        self.skmodel = svm.SVR(verbose=verbosity)
        self.skmodel = self.grid_search(self.skmodel)

    def grid_search(self, model):
        # Radial basis function grid
        rbf_grid = {'kernel': ['rbf'],
                    'C': [1, 10, 100, 1000],
                    'epsilon': logspace(0, 1, 5),
                    'gamma': logspace(0, 1, 5)}

        # Polynomial function grid
        poly_grid = {'kernel': ['poly'],
                      'C': [1, 10, 100, 1000],
                      'degree': [1, 2, 3, 4, 5],
                      'coef0': logspace(0, 1, 5),
                      'gamma': logspace(0, 1, 5)}
        # Linear function grid
        linear_grid = {'kernel': ['linear'],
                       'C': [1, 10, 100, 1000]}

        # Sigmoid function grid
        sigmoid_grid = {'kernel': ['sigmoid'],
                        'C': [1, 10, 100, 1000],
                        'coef0': logspace(0, 1, 5),
                        'gamma': logspace(0, 1, 5)}

        param_grid = [rbf_grid, poly_grid, linear_grid, sigmoid_grid]
        return GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=1)


class SupportVectorClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names, model_type='classification')
        self.skmodel = svm.SVC(verbose=verbosity, kernel='poly', degree=2, C=600000)
        self.skmodel = self.grid_search(self.skmodel)

    def grid_search(self, model):
        # Radial basis function grid
        rbf_grid = {'kernel': ['rbf'],
                    'C': [1, 10, 100, 1000],
                    'epsilon': np.logspace(0, 1, 5),
                    'gamma': np.logspace(0, 1, 5)}

        # Polynomial function grid
        poly_grid = {'kernel': ['poly'],
                     'C': [1, 10, 100, 1000],
                     'degree': [1, 2, 3, 4, 5],
                     'coef0': np.logspace(0, 1, 5),
                     'gamma': np.logspace(0, 1, 5)}

        # Linear function grid
        linear_grid = {'kernel': ['linear'], 'C': [0.1, 1, 10, 100], 'epsilon': np.logspace(0, 1, 5)}

        # Sigmoid function grid
        sigmoid_grid = {'kernel': ['sigmoid'],
                        'C': [1, 10, 100, 1000],
                        'coef0': np.logspace(0, 1, 5),
                        'epsilon': np.logspace(0, 1, 5),
                        'gamma': np.logspace(0, 1, 5)}

        simple_grid = {'kernel': ['poly'], 'degree': [2], 'C': [6000000]}

        param_grid = [simple_grid]

        param_grid = [rbf_grid, poly_grid, linear_grid, sigmoid_grid]
        return GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=1)

    def variable_to_validate(self):
        return 'degree'

    def predict_for_roc(self, x_data):
        return self.skmodel.decision_function(x_data)
