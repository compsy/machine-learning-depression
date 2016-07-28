from sklearn.metrics import mean_squared_error
from data_output.std_logger import L
from machine_learning_evaluation.evaluation import Evaluation
import numpy as np


class VarianceEvaluation(Evaluation):

    def __init__(self,
                 name='Variance Evaluator',):
        super().__init__(name=name, model_type='regression')

    def evaluate(self, y_true, y_predicted):
        return (np.var(y_true), np.var(y_predicted))

    def print_evaluation(self, model, y_true, y_pred):
        true_variance, predicted_variance = self.evaluate(y_true, y_pred)
        L.info("\t -> %s of %s: true data: %0.2f, predicted data: %0.2f" %
               (self.name, model.given_name, true_variance, predicted_variance))
