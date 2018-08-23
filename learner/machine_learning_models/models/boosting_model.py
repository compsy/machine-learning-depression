from scipy.stats import halflogistic, randint
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor, GradientBoostingClassifier
from learner.machine_learning_models.machine_learning_model import MachineLearningModel
import numpy as np


class BoostingModel(MachineLearningModel):

    def __init__(self, x, y, y_names, grid_search, verbosity, **kwargs):

        hyperparameters = {
            'n_estimators': 20,
            'max_depth': 100,
            'learning_rate': 0.1,
            'min_samples_split': 5,
            'min_samples_leaf': 5,
            'max_features': 'auto',
            'verbose': verbosity
        }

        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, verbosity=verbosity, pretty_name = 'Gradient Boosting', model_type='regression', **kwargs)
        self.skmodel = GradientBoostingRegressor(**self.hyperparameters)

        if grid_search:
            parameter_grid = {
                'n_estimators': np.unique(np.round(np.logspace(0, 2, 20))),
                'max_depth': np.unique(np.round(np.logspace(0, 2, 20))),
                'learning_rate': np.logspace(-2, 1, 10),
                'min_samples_split': np.unique(np.round(np.logspace(0, 1, 10))),
                'min_samples_leaf': np.unique(np.round(np.logspace(0, 1, 10))),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }

            random_parameter_grid = {
                'n_estimators': randint(1, 100),
                'max_depth': randint(1, 100),
                'learning_rate': halflogistic(),
                'n_estimators': halflogistic(scale=100),
                'max_depth': halflogistic(scale=100),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            self.grid_search([parameter_grid], [random_parameter_grid])


class BoostingClassificationModel(MachineLearningModel):

    def __init__(self, x, y, y_names, grid_search, verbosity, **kwargs):
        hyperparameters = {'n_estimators': 1000, 'max_depth': 5, 'verbose': verbosity}

        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, verbosity=verbosity, pretty_name = 'Gradient Boosting', model_type='classification', **kwargs)
        self.skmodel = GradientBoostingClassifier(**self.hyperparameters)

        if grid_search:
            parameter_grid = {
                'n_estimators': np.unique(np.round(np.logspace(0, 2, 20))),
                'max_depth': np.unique(np.round(np.logspace(0, 2, 20))),
                'learning_rate': np.logspace(-2, 1, 10),
                'min_samples_split': np.unique(np.round(np.logspace(0, 2, 10))),
                'min_samples_leaf': np.unique(np.round(np.logspace(0, 1, 10))),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }

            random_parameter_grid = {
                'n_estimators': randint(1, 1000),
                'max_depth': randint(1, 1000),
                'learning_rate': halflogistic(),
                'min_samples_split': randint(2, 1000),
                'min_samples_leaf': randint(1, 1000),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            self.grid_search([parameter_grid], [random_parameter_grid])
