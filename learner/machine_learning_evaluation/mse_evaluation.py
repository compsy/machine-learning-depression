from sklearn.metrics import mean_squared_error
from ..machine_learning_evaluation.evaluation import Evaluation


class MseEvaluation(Evaluation):

    def __init__(self):
        super().__init__(name='MSE Evaluator', model_type='regression')

    def evaluate(self, y_true, y_predicted):
        return mean_squared_error(y_true, y_predicted)
