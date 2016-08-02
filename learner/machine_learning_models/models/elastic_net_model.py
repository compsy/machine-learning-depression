from machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn import linear_model
from sklearn.linear_model import ElasticNet

from data_output.std_logger import L
from machine_learning_models.models.boosting_model import BoostingClassificationModel
import numpy as np

class ElasticNetModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True):
        super().__init__(x, y, x_names, y_names, model_type='classification')
        self.skmodel = ElasticNet(alpha=1,
                                  l1_ratio=0.5,
                                  max_iter=10000)

        if grid_search:
            parameter_grid = {
                    'alpha': np.logspace(-10, 3, 100),
                    'l1_ratio': np.logspace(-10, 0, 100)
                    }
            self.grid_search([parameter_grid])

    def predict_for_roc(self, x_data):
        # TODO: What is the difference between log_proba and proba?
        L.info(self.skmodel.coef_)

        L.info(self.skmodel.sparse_coef_)
        return self.skmodel.predict_log_proba(x_data)[:, 1]
