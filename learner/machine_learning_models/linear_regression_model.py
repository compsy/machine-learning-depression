from .machine_learning_model import MachineLearningModel

from sklearn import linear_model


class LinearRegressionModel(MachineLearningModel):

    def train(self):
        if (self.skmodel is not None):
            return self

        self.skmodel = linear_model.Lasso(alpha=0.01, fit_intercept=False, normalize=True, copy_X=False)
        self.skmodel = self.skmodel.fit(self.x_train, self.y_train)
        return self
