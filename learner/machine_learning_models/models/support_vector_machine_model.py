from sklearn.grid_search import GridSearchCV

from machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn import svm
from numpy import logspace
import numpy as np


class SupportVectorRegressionModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='regression', **kwargs)
        self.skmodel = svm.SVR(verbose=verbosity)
        # Radial basis function grid
        rbf_grid = {'kernel': ['rbf'],
                    'C': [1, 10, 100, 1000],
                    'epsilon': logspace(0, 1, 5),
                    'gamma': logspace(0, 1, 5)}

        # Polynomial function grid
        poly_grid = {'kernel': ['poly'],
                     'C': [1, 10, 100, 1000],
                     'degree': [2, 3, 4, 5],
                     'coef0': logspace(0, 1, 5),
                     'gamma': logspace(0, 1, 5)}
        # Linear function grid
        linear_grid = {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

        # Sigmoid function grid
        sigmoid_grid = {'kernel': ['sigmoid'],
                        'C': [1, 10, 100, 1000],
                        'coef0': logspace(0, 1, 5),
                        'gamma': logspace(0, 1, 5)}

        param_grid = [rbf_grid, poly_grid, linear_grid, sigmoid_grid]
        param_grid = [rbf_grid, poly_grid, sigmoid_grid]
        self.grid_search(param_grid)


class SupportVectorClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='classification', **kwargs)
        self.skmodel = svm.SVC(verbose=verbosity, kernel='poly', degree=2, C=600000)

        # Radial basis function grid
        rbf_grid = {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': np.logspace(0, 1, 5)}

        # Polynomial function grid
        poly_grid = {'kernel': ['poly'],
                     'C': [1, 10, 100, 1000],
                     'degree': [2, 3, 4, 5],
                     'coef0': np.logspace(0, 1, 5),
                     'gamma': np.logspace(0, 1, 5)}

        # Linear function grid
        linear_grid = {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}

        # Sigmoid function grid
        sigmoid_grid = {'kernel': ['sigmoid'],
                        'C': [1, 10, 100, 1000],
                        'coef0': np.logspace(0, 1, 5),
                        'gamma': np.logspace(0, 1, 5)}

        #param_grid = [poly_grid, rbf_grid, linear_grid, sigmoid_grid]
        param_grid = [poly_grid, rbf_grid, sigmoid_grid]
        # param_grid = [poly_grid]
        self.grid_search(param_grid)

    def variable_to_validate(self):
        return 'degree'
