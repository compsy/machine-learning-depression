from machineLearningModels.MachineLearningModel import MachineLearningModel
from sklearn import cross_validation

from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model


class LinearRegressionModel(MachineLearningModel):

    def train(self):
        lr = linear_model.LinearRegression()

        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validated:
        return cross_val_predict(lr, self.x, self.y, cv=10)
