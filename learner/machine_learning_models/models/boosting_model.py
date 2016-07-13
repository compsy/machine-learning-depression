from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor, GradientBoostingClassifier

from machine_learning_models.machine_learning_model import MachineLearningModel


class BoostingModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names)
        self.skmodel = GradientBoostingRegressor(verbose=verbosity)


class BoostingClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names)
        self.skmodel = GradientBoostingClassifier(verbose=verbosity)

    def predict_for_roc(self, x_data):
        return self.skmodel.decision_function(x_data)