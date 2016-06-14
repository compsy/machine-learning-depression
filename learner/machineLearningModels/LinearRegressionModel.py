from machineLearningModels.MachineLearningModel import MachineLearningModel
from sklearn import cross_validation

from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model


class LinearRegressionModel(MachineLearningModel):

    def train(self):
        model = linear_model.LinearRegression()
        model = model.fit(self.x, self.y)
        return model

    def validate(self):
        model = self.train()
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validated:
        return cross_val_predict(model, self.x, self.y, cv=4)
