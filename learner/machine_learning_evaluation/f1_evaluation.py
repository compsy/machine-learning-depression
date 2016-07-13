from sklearn.metrics import f1_score
from learner.machine_learning_evaluation.evaluation import Evaluation


class F1Evaluation(Evaluation):

    def __init__(self):
        super().__init__(name='F1 Evaluator', model_type='classification')

    def evaluate(self, y_true, y_pred):
        return f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
