from sklearn.metrics import explained_variance_score
from machine_learning_evaluation.evaluation import Evaluation


class ExplainedVarianceEvaluation(Evaluation):

    def __init__(self):
        super().__init__(name='Explained Variance Evaluator', model_type='regression')

    def evaluate(self, y_true, y_pred):
        return explained_variance_score(y_true, y_pred)
