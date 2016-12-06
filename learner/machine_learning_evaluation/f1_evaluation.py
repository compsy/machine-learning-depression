from sklearn.metrics import f1_score
from machine_learning_evaluation.evaluation import Evaluation


class F1Evaluation(Evaluation):

    def __init__(self, pos_label = 0):
        super().__init__(name='F1 Evaluator', model_type='classification')
        self.pos_label = pos_label

    def evaluate(self, y_true, y_pred):
        return f1_score(y_true=y_true, y_pred=y_pred, pos_label=self.pos_label)
