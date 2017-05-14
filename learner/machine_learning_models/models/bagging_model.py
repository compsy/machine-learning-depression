from sklearn.ensemble.bagging import BaggingRegressor, BaggingClassifier

class BaggingModel:

    @staticmethod
    def use_bagging(verbosity, skmodel):
        return BaggingRegressor(
            base_estimator=skmodel, verbose=verbosity, n_estimators=100, bootstrap=True, max_samples=100)


class BaggingClassificationModel:

    @staticmethod
    def use_bagging(verbosity, skmodel):
        return BaggingClassifier(
            base_estimator=skmodel, verbose=verbosity, n_estimators=100, bootstrap=True, max_samples=100)
