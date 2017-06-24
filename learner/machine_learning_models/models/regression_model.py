from sklearn.linear_model import LogisticRegression
from learner.machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn.linear_model import ElasticNetCV, ElasticNet
from learner.data_output.std_logger import L
from scipy.stats import expon, halflogistic, uniform
import numpy as np


class ElasticNetModel(MachineLearningModel):

    def __init__(self, x, y, y_names, verbosity, grid_search, **kwargs):
        hyperparameters = {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 10000}
        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, verbosity=verbosity, model_type='classification', **kwargs)
        # TODO: Change to elasticnet CV
        self.skmodel = ElasticNet(**self.hyperparameters)

        if grid_search:
            parameter_grid = {'alpha': np.logspace(-10, 3, 100), 'l1_ratio': np.logspace(-10, 0, 100)}

            random_parameter_grid = {
                # Uniformely between 0-1
                'alpha': uniform(),
                'l1_ratio': uniform()
            }
            self.grid_search([parameter_grid], [random_parameter_grid])


class LogisticRegressionModel(MachineLearningModel):

    def __init__(self, x, y, y_names, verbosity, grid_search, **kwargs):
        hyperparameters = {
            'penalty': 'l2',
            'C': 0.1,
            'verbose': verbosity,
            'random_state': 42,
            'n_jobs': -1,
            'tol': 0.0001,
        }
        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, verbosity=verbosity, model_type='classification', **kwargs)
        self.skmodel = LogisticRegression(**self.hyperparameters)

        if grid_search:
            parameter_grid = {'penalty': ['l1', 'l2'], 'C': np.logspace(-10, 2, 5)}
            random_parameter_grid = {'penalty': ['l1', 'l2'], 'C': halflogistic(10)}
            self.grid_search([parameter_grid], [random_parameter_grid])
