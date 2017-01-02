from sklearn.metrics import accuracy_score
from learner.machine_learning_evaluation.evaluation import Evaluation


class AccuracyEvaluation(Evaluation):

    def __init__(self):
        super().__init__(name='Accuracy Evaluator', model_type='classification')

    def evaluate(self, y_true, y_pred):
        """
        Prints the accuracy of a model using crossvalidation on the test set
        """
        return accuracy_score(y_true= y_true, y_pred=y_pred)
