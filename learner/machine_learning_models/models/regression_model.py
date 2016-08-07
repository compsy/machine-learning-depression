from scipy.stats import halflogistic

from machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from machine_learning_models.models.boosting_model import BoostingClassificationModel
import numpy as np


class LinearRegressionModel(MachineLearningModel):
    MAX_ITERATIONS = 10000

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='regression', **kwargs)
        self.skmodel = linear_model.LassoCV(eps=1e-2,
                                            n_alphas=300,
                                            fit_intercept=True,
                                            copy_X=False,
                                            max_iter=self.MAX_ITERATIONS,
                                            verbose=verbosity)


class LogisticRegressionModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True, **kwargs):
        super().__init__(x, y, x_names, y_names, verbosity=verbosity, model_type='classification', **kwargs)
        self.skmodel = LogisticRegression(penalty='l2',
                                          C=0.1,
                                          verbose=verbosity,
                                          random_state=42,
                                          tol=0.000001,
                                          max_iter=100000)

        if grid_search:
            parameter_grid = {'penalty': ['l1', 'l2'], 'C': np.logspace(0, 2, 5)}
            random_parameter_grid = {'penalty': ['l1', 'l2'], 'C': halflogistic(10)}
            self.grid_search([parameter_grid], [random_parameter_grid])
