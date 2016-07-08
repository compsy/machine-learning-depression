from machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn.cross_validation import cross_val_predict
from sklearn.tree import DecisionTreeRegressor


class RegressionTreeModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names)
        self.skmodel = DecisionTreeRegressor(max_depth=5)

