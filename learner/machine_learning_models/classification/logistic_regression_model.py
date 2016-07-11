from machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names, model_type='classification')
        self.skmodel = LogisticRegression()

    def predict_for_roc(self, x_data):
        return self.skmodel.decision_function(x_data)
