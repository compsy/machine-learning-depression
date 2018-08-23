from scipy.stats import halflogistic, randint, uniform
from sklearn.neural_network import MLPClassifier

from learner.machine_learning_models.machine_learning_model import MachineLearningModel
import numpy as np

from learner.machine_learning_models.models.boosting_model import BoostingClassificationModel
from scipy.stats import expon

class NeuralNetworkModel(MachineLearningModel):

    def __init__(self, x, y, y_names, grid_search, verbosity, **kwargs):
        hyperparameters = {
            'hidden_layer_sizes':  (100,),
            'activation':          'relu',
            'solver':              'adam',
            'alpha':               0.0001,
            'batch_size':          'auto',
            'learning_rate':       'constant',
            'learning_rate_init':  0.001,
            'max_iter':            200,
            'shuffle':             True,
            'random_state':        None,
            'tol':                 1e-4,
            'verbose':             False,
            'warm_start':          False,
            'momentum':            0.9,
            'nesterovs_momentum':  True,
            'early_stopping':      False,
            'validation_fraction': 0.1,
            'epsilon':             1e-8

        }
        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, pretty_name = 'Neural Network', model_type='classification', verbosity=verbosity, **kwargs)

        self.skmodel = MLPClassifier(**self.hyperparameters)

        if grid_search:
            parameter_grid = {
                'hidden_layer_sizes':  [(100,),(100,10)],
                'activation':          ['logistic', 'relu'],
                'solver':              ['lbfgs', 'sgd', 'adam'],
                'alpha':               halflogistic(scale=0.0001),
                'batch_size':          ['auto'],
                'learning_rate':       ['invscaling', 'adaptive', 'constant'],
                'learning_rate_init':  halflogistic(scale=0.001),
                'power_t':             uniform(loc=0, scale=1),
                'max_iter':            [400],
                'shuffle':             [True],
                'random_state':        [None],
                'tol':                 halflogistic(scale=1e-4),
                'verbose':             [False],
                'warm_start':          [False],
                'momentum':            uniform(loc=0.5, scale=0.4),
                'nesterovs_momentum':  [True],
                'early_stopping':      [False],
                'validation_fraction': [0.1],
                'epsilon':             [1e-8]
            }
            random_parameter_grid = {
                'hidden_layer_sizes':  [(100,)],
                'activation':          ['logistic', 'relu'],
                'solver':              ['lbfgs', 'sgd', 'adam'],
                'alpha':               halflogistic(scale=0.0001),
                'batch_size':          ['auto'],
                'learning_rate':       ['invscaling', 'adaptive', 'constant'],
                'learning_rate_init':  halflogistic(scale=0.001),
                'power_t':             uniform(loc=0, scale=1),
                'max_iter':            [400],
                'shuffle':             [True],
                'random_state':        [None],
                'tol':                 halflogistic(scale=1e-4),
                'verbose':             [False],
                'warm_start':          [False],
                'momentum':            uniform(loc=0.5, scale=0.4),
                'nesterovs_momentum':  [True],
                'early_stopping':      [False],
                'validation_fraction': [0.1],
                'epsilon':             [1e-8]
            }
            random_parameter_grid = {
                'tol':                 halflogistic(scale=1e-4),
            }
            self.grid_search([parameter_grid], [random_parameter_grid])
