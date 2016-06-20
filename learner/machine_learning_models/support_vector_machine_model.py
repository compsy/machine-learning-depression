from sklearn.grid_search import GridSearchCV

from .machine_learning_model import MachineLearningModel
from sklearn import svm
from numpy import logspace


class SupportVectorMachineModel(MachineLearningModel):

    def train(self):
        self.skmodel = self.train_with_grid_search()
        return self

    def train_with_grid_search(self):
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
        linear_grid = {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'epsilon': logspace(0, 1, 5)}

        # Sigmoid function grid
        # sigmoid_grid = {'kernel': ['sigmoid'],
        #                 'C': [1, 10, 100, 1000],
        #                 'coef0': logspace(0, 1, 5),
        #                 'epsilon': logspace(0, 1, 5),
        #                 'gamma': logspace(0, 1, 5)}

        param_grid = [linear_grid]
        skmodel = svm.SVR()
        skmodel = GridSearchCV(estimator=skmodel, param_grid=param_grid, n_jobs=-1, verbose=1)
        return skmodel
