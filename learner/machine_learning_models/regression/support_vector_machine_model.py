from sklearn.grid_search import GridSearchCV

from machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn import svm
from numpy import logspace


class SupportVectorMachineModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names)
        self.skmodel = svm.SVR(verbose=verbosity)

    def train_not_yet_used(self):
        # Radial basis function grid
        # rbf_grid = {'kernel': ['rbf'],
        #             'C': [1, 10, 100, 1000],
        #             'epsilon': logspace(0, 1, 5),
        #             'gamma': logspace(0, 1, 5)}

        # Polynomial function grid
        # poly_grid = {'kernel': ['poly'],
        #              'C': [1, 10, 100, 1000],
        #              'degree': [1, 2, 3, 4, 5],
        #              'coef0': logspace(0, 1, 5),
        #              'epsilon': logspace(0, 1, 5),
        #              'gamma': logspace(0, 1, 5)}

        # Linear function grid
        #linear_grid = {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'epsilon': logspace(0, 1, 5)}

        # Sigmoid function grid
        # sigmoid_grid = {'kernel': ['sigmoid'],
        #                 'C': [1, 10, 100, 1000],
        #                 'coef0': logspace(0, 1, 5),
        #                 'epsilon': logspace(0, 1, 5),
        #                 'gamma': logspace(0, 1, 5)}

        param_grid = [linear_grid]
        self.skmodel = GridSearchCV(estimator=self.skmodel, param_grid=param_grid, n_jobs=-1, verbose=1)
        return self.skmodel
