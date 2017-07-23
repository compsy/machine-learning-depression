from sklearn.metrics import cohen_kappa_score
from learner.machine_learning_evaluation.evaluation import Evaluation


class KappaEvaluation(Evaluation):

    def __init__(self):
        super().__init__(name='Kappa Evaluator', model_type='classification')

    def evaluate(self, y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred)
