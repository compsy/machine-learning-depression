from machineLearningModels.machine_learning_model import MachineLearningModel
from sklearn.cross_validation import cross_val_predict
from sklearn.tree import DecisionTreeRegressor


class RegressionTreeModel(MachineLearningModel):

    def train(self):
        if (self.skmodel is not None):
            return self

        self.skmodel = DecisionTreeRegressor(max_depth=1000)
        self.skmodel = self.skmodel.fit(self.x, self.y)
        return self
