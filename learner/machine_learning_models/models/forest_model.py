from scipy.stats import halflogistic, randint

from learner.machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class RandomForestClassificationModel(MachineLearningModel):

    def __init__(self, x, y, y_names, grid_search, verbosity, **kwargs):
        hyperparameters = {
            'n_estimators': 5,
            'max_depth': 10,
            'max_features': 'auto',
        }
        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, pretty_name = 'Random Forest',verbosity=verbosity, model_type='classification', **kwargs)
        self.skmodel = RandomForestClassifier(**self.hyperparameters)

        if grid_search:
            parameter_grid = {
                'n_estimators': np.logspace(0, 2, 20),
                'max_depth': np.logspace(0, 2, 20),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            random_parameter_grid = {
                'n_estimators': randint(1, 10),
                'max_depth': randint(1, 100),
                'max_features': ['auto', 'sqrt', 'log2', None]
            }
            self.grid_search([parameter_grid], [random_parameter_grid])
