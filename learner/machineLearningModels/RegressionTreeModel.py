from machineLearningModels.MachineLearningModel import MachineLearningModel
from sklearn.cross_validation import cross_val_predict
from sklearn.tree import DecisionTreeRegressor


class RegressionTreeModel(MachineLearningModel):

    def train(self):
        regression_tree = DecisionTreeRegressor(max_depth=10)
        return cross_val_predict(regression_tree, self.x, self.y, cv=10)
