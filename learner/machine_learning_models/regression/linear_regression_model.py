from machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn import linear_model


class LinearRegressionModel(MachineLearningModel):
    MAX_ITERATIONS = 10000

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names)
        self.skmodel = linear_model.LassoCV(eps=1e-6, n_alphas=3000,
                                            fit_intercept=False, normalize=True, copy_X=False,
                                            max_iter=self.MAX_ITERATIONS, verbose=verbosity, n_jobs=-1)