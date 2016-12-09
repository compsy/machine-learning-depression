from scipy.stats import halflogistic, randint


from learner.machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='classification', verbosity=verbosity, **kwargs)
        self.skmodel = RandomForestClassifier(n_estimators=5)

        if grid_search:
            parameter_grid = {
                'n_estimators': np.logspace(0, 2, 20),
                'max_depth': np.logspace(0, 2, 20),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            random_parameter_grid = {
                'n_estimators': randint(1,100),
                'max_depth': randint(1,100),
                'max_features': ['auto', 'sqrt', 'log2', None]
            }
            self.grid_search([parameter_grid], [random_parameter_grid])

