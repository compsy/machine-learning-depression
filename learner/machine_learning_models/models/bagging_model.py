from sklearn.ensemble.bagging import BaggingRegressor, BaggingClassifier

from machine_learning_models.machine_learning_model import MachineLearningModel


class BaggingModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names, model_type='regression')
        self.skmodel = BaggingRegressor(verbose=verbosity, n_estimators=100, bootstrap=True, max_samples=100)


class BaggingClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names)
        self.skmodel = BaggingClassifier(verbose=verbosity, n_estimators=100, bootstrap=True, max_samples=100)

    def predict_for_roc(self, x_data):
        return self.skmodel.predict_log_proba(x_data)[:, 1]
