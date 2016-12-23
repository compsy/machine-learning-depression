from scipy.stats import expon, halflogistic, uniform

from learner.machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn.naive_bayes import GaussianNB, BernoulliNB
import numpy as np


class NaiveBayesModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, **kwargs)


class GaussianNaiveBayesModel(NaiveBayesModel):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, verbosity=verbosity, model_type='classification', **kwargs)
        self.skmodel = GaussianNB()


class BernoulliNaiveBayesModel(NaiveBayesModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True, **kwargs):
        super().__init__(x, y, x_names, y_names, verbosity=verbosity, model_type='classification', **kwargs)
        self.skmodel = BernoulliNB()

        if grid_search:
            parameter_grid = {
                'alpha': np.logspace(-10, 3, 100),
                'binarize': np.logspace(-10, 0, 100),
                'fit_prior': [True, False]
            }

            random_parameter_grid = {'alpha': halflogistic(), 'binarize': halflogistic(), 'fit_prior': [True, False]}
            self.grid_search([parameter_grid], [random_parameter_grid])
