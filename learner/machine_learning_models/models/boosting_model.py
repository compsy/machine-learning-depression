from scipy.stats import halflogistic
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor, GradientBoostingClassifier
from machine_learning_models.machine_learning_model import MachineLearningModel
import numpy as np


class BoostingModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='regression', **kwargs)
        self.skmodel = GradientBoostingRegressor(verbose=verbosity)
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
                'n_estimators': halflogistic(scale=100),
                'max_depth': halflogistic(scale=100),
                'learning_rate': halflogistic(),
                'min_samples_split': halflogistic(scale=100),
                'min_samples_leaf': halflogistic(scale=100),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            self.grid_search([parameter_grid], [random_parameter_grid])


class BoostingClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='classification', **kwargs)
        self.skmodel = GradientBoostingClassifier(verbose=verbosity, n_estimators=1000, max_depth=5)
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
                'n_estimators': halflogistic(scale=100),
                'max_depth': halflogistic(scale=100),
                'learning_rate': halflogistic(),
                'min_samples_split': halflogistic(scale=100),
                'min_samples_leaf': halflogistic(scale=100),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            self.grid_search([parameter_grid], [random_parameter_grid])