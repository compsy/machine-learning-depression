from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor, GradientBoostingClassifier
from machine_learning_models.machine_learning_model import MachineLearningModel
import numpy as np


class BoostingModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        self.skmodel = GradientBoostingRegressor(verbose=verbosity)
        super().__init__(x, y, x_names, y_names, model_type='regression')


class BoostingClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True):
        super().__init__(x, y, x_names, y_names, model_type='classification')
        self.skmodel = GradientBoostingClassifier(verbose=verbosity, n_estimators=1000, max_depth=5)
        if grid_search:
            parameter_grid = {
                'max_depth': np.logspace(0, 2, 20),
                'learning_rate': np.logspace(0, 1, 10),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            self.grid_search([parameter_grid])

    def predict_for_roc(self, x_data):
        return self.skmodel.decision_function(x_data)
