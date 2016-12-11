from learner.machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn.naive_bayes import GaussianNB, BernoulliNB


class NaiveBayesModel(MachineLearningModel):
    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, **kwargs)

class GaussianNaiveBayesModel(NaiveBayesModel):
    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, verbosity = verbosity, model_type='classification', **kwargs)
        self.skmodel = GaussianNB()

class BernoulliNaiveBayesModel(NaiveBayesModel):
    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, verbosity = verbosity, model_type='classification', **kwargs)
        self.skmodel = BernoulliNB()