from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

from machine_learning_models.machine_learning_model import MachineLearningModel


class BoostingModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names)
        self.skmodel = GradientBoostingRegressor(verbose=verbosity)
