from scipy.stats import expon, halflogistic, uniform
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import numpy as np
from learner.machine_learning_models.machine_learning_model import MachineLearningModel


class StochasticGradientDescentClassificationModel(MachineLearningModel):

    def __init__(self, x, y, y_names, grid_search, verbosity, **kwargs):
        hyperparameters = {
            'alpha': 0.5,
            'average': False,
            'class_weight': None,
            'epsilon': 0.1,
            'eta0': 0.0,
            'fit_intercept': True,
            'l1_ratio': 0.5,
            'learning_rate': 'optimal',
            'loss': 'log',
            'shuffle': True,
            'verbose': verbosity
        }

        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, pretty_name = 'Stochastic Gradient Descent Estimator', verbosity=verbosity, model_type='classification', **kwargs)
        self.skmodel = SGDClassifier(**self.hyperparameters)

        # Radial basis function grid
        grid = {
            'loss': ['log'],
            'alpha': np.logspace(-10, 3, 100),
            'average': [True, False],
            'class_weight': ['balanced', None],
            'epsilon': np.logspace(-10, 3, 100),
            'eta0': np.logspace(-10, 3, 100),
            'l1_ratio': np.logspace(-10, 0, 100),
        }

        random_parameter_grid = {
            'loss': ['log'],
            'alpha': halflogistic(scale=.1),
            'average': [True, False],
            'class_weight': ['balanced', None],
            'epsilon': halflogistic(scale=.1),
            'eta0': halflogistic(scale=.1),
            'l1_ratio': uniform()
        }

        self.exhaustive_param_grid = [grid]
        self.random_param_grid = [random_parameter_grid]
        if grid_search:
            self.grid_search(self.exhaustive_param_grid, self.random_param_grid)
