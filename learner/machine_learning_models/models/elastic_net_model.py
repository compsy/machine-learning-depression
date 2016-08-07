from machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn.linear_model import ElasticNetCV, ElasticNet
from data_output.std_logger import L
import numpy as np
from scipy.stats import expon

class ElasticNetModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='classification', **kwargs)
        # TODO: Change to elasticnet CV
        self.skmodel = ElasticNet(alpha=0.1,
                                  l1_ratio=0.5,
                                  max_iter=10000)

        if grid_search:
            parameter_grid = {
                    'alpha': np.logspace(-10, 3, 100),
                    'l1_ratio': np.logspace(-10, 0, 100)
                    }

            random_parameter_grid = {
                'alpha': expon(scale=100),
                'l1_ratio': expon(scale=.1)
            }
            self.grid_search([parameter_grid], [random_parameter_grid])

    def determine_best_variables(self, top=25):
        if self.was_trained:
            assert len(self.skmodel.coef_) == len(self.x_names)
            L.info('The most predictive variables are:')
            indices = self.skmodel.sparse_coef_.indices
            data = self.skmodel.sparse_coef_.data
            zipped = list(zip(data, indices))
            zipped.sort(reverse=True, key=lambda tup: abs(tup[0]))
            i = 0
            for coefficient, index in zipped:
                i+=1
                var_name = self.x_names[index]
                L.info('--> %d\t%0.5f\t%s' % (i, coefficient, var_name))
                if(i>=top): break

            return zipped
