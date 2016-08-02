from machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn import linear_model
from sklearn.linear_model import ElasticNet

from machine_learning_models.models.boosting_model import BoostingClassificationModel
import numpy as np

class ElasticNetModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True):
        super().__init__(x, y, x_names, y_names, model_type='classification')
        self.skmodel = ElasticNet(alpha=1,
                                  l1_ratio=0.5)

        if grid_search:
            parameter_grid = {
                    'alpha': np.logspace(0, 3, 50),
                    'l1_ratio': np.logspace(-1, 1, 50)
                    }
            self.grid_search([parameter_grid])
