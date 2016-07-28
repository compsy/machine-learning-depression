from sklearn.metrics import mean_squared_error
from machine_learning_evaluation.evaluation import Evaluation
import numpy as np


class MseEvaluation(Evaluation):

    def __init__(self,
                 name='MSE Evaluator',):
        super().__init__(name=name, model_type='regression')

    def evaluate(self, y_true, y_predicted):
        return mean_squared_error(y_true, y_predicted)


class RootMseEvaluation(MseEvaluation):

    def __init__(self):
        super().__init__(name='Root MSE Evaluator')

    def evaluate(self, y_true, y_predicted):
        return np.sqrt(super(RootMseEvaluation, self).evaluate(y_true, y_predicted))
