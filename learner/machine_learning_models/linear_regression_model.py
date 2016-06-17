from .machine_learning_model import MachineLearningModel
from sklearn import linear_model


class LinearRegressionModel(MachineLearningModel):

    def train(self):
        if (self.skmodel is not None):
            return self

        self.skmodel = linear_model.LinearRegression(fit_intercept=True,
                                                     normalize=True,
                                                     copy_X=False,
                                                     n_jobs=-1)
        self.skmodel = self.skmodel.fit(self.x, self.y)
        return self
