import math
from sklearn.metrics import recall_score
from learner.machine_learning_evaluation.evaluation import Evaluation
from imblearn.metrics import geometric_mean_score


class GeometricMeanEvaluation(Evaluation):

    def __init__(self, pos_label=0):
        super().__init__(name='Geometric Mean Evaluator', model_type='classification')
        self.pos_label = pos_label

    def evaluate(self, y_true, y_pred):
        # tpr = recall_score(y_true=y_true, y_pred=y_pred, pos_label=self.pos_label)
        # neg_label = 0 if self.pos_label == 1 else 1
        # tnr = recall_score(y_true=y_true, y_pred=y_pred, pos_label=neg_label)
        # gmean = math.sqrt(tpr * tnr)
        gmean = geometric_mean_score(y_true, y_pred, pos_label=self.pos_label)

        return gmean

if __name__ == '__main__':
    d = GeometricMeanEvaluation()
    true = [1,1,0,1,0,0]
    pred = [1,1,0,1,0,0]
    print(d.evaluate(true, pred))
