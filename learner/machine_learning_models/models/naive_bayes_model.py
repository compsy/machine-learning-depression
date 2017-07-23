from scipy.stats import expon, halflogistic, uniform

from learner.machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn.naive_bayes import GaussianNB, BernoulliNB
import numpy as np


class NaiveBayesModel(MachineLearningModel):

    def __init__(self, x, y, y_names, verbosity, **kwargs):
        super().__init__(x, y, y_names, verbosity=verbosity, **kwargs)


class GaussianNaiveBayesModel(NaiveBayesModel):

    def __init__(self, x, y, y_names, grid_search, verbosity, **kwargs):
        hyperparameters = {'verbosity': verbosity}
        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, verbosity=verbosity, pretty_name = 'Gaussian Naive Bayes',  model_type='regression', **kwargs)
        self.skmodel = GaussianNB()


class BernoulliNaiveBayesModel(NaiveBayesModel):

    def __init__(self, x, y, y_names, verbosity, grid_search, **kwargs):
        hyperparameters = {'alpha': 0.1, 'binarize': 0.5, 'fit_prior': True}
        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, verbosity=verbosity, pretty_name = 'Bernoulli Naive Bayes', model_type='classification', **kwargs)

        self.skmodel = BernoulliNB(**self.hyperparameters)

        if grid_search:
            parameter_grid = {
                'alpha': np.logspace(-10, 3, 100),
                'binarize': np.logspace(-10, 0, 100),
                'fit_prior': [True, False]
            }

            random_parameter_grid = {'alpha': halflogistic(), 'binarize': halflogistic(), 'fit_prior': [True, False]}
            self.grid_search([parameter_grid], [random_parameter_grid])
