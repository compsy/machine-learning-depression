from sklearn.metrics import roc_auc_score
from learner.machine_learning_evaluation.evaluation import Evaluation


class AucEvaluation(Evaluation):

    def __init__(self):
        super().__init__(name='AUC Evaluator', model_type='classification')

    def evaluate(self, y_true, y_score):
        """
        Prints the auc of a model using the test set
        """
        return roc_auc_score(y_true=y_true, y_score=y_score)
