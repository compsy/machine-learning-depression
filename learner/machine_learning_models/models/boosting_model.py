from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor, GradientBoostingClassifier

from machine_learning_models.machine_learning_model import MachineLearningModel


class BoostingModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        self.skmodel = GradientBoostingRegressor(verbose=verbosity, init=self.skmodel)
        super().__init__(x, y, x_names, y_names, model_type='regression')


class BoostingClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        self.skmodel = GradientBoostingClassifier(verbose=verbosity, init=self.skmodel,
                                                  n_estimators=100,
                                                  learning_rate=0.01,
                                                  max_depth=50)
        super().__init__(x, y, x_names, y_names, **kwargs)

    def predict_for_roc(self, x_data):
        return self.skmodel.decision_function(x_data)
