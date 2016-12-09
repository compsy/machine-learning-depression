from sklearn.ensemble.bagging import BaggingRegressor, BaggingClassifier

from learner.machine_learning_models.machine_learning_model import MachineLearningModel


class BaggingModel:

    @staticmethod
    def use_bagging(verbosity,skmodel):
        return BaggingRegressor(base_estimator=skmodel,
                                        verbose=verbosity,
                                        n_estimators=100,
                                        bootstrap=True,
                                        max_samples=100)


class BaggingClassificationModel(MachineLearningModel):

    @staticmethod
    def use_bagging(verbosity, skmodel):
        return BaggingClassifier(base_estimator=skmodel,
                                         verbose=verbosity,
                                         n_estimators=100,
                                         bootstrap=True,
                                         max_samples=100)


