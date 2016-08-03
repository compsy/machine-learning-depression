from sklearn.ensemble.bagging import BaggingRegressor, BaggingClassifier

from machine_learning_models.machine_learning_model import MachineLearningModel


class BaggingModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='regression', **kwargs)
        self.skmodel = BaggingRegressor(base_estimator=self.skmodel,
                                        verbose=verbosity,
                                        n_estimators=100,
                                        bootstrap=True,
                                        max_samples=100)


class BaggingClassificationModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='classification', **kwargs)
        self.skmodel = BaggingClassifier(base_estimator=self.skmodel,
                                         verbose=verbosity,
                                         n_estimators=100,
                                         bootstrap=True,
                                         max_samples=100)
