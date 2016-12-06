from learner.machine_learning_evaluation.variance_evaluation import VarianceEvaluation
import pytest
import numpy as np

import logging

class TestVarianceEvaluation():
    @pytest.fixture()
    def subject(self):
        subject = VarianceEvaluation()
        return subject

    def test_mse_initialization(self, subject):
        assert subject.name == 'Variance Evaluator'
        assert subject.model_type == 'regression'

    def test_evaluate_returns_the_variance_of_the_true_and_pred_data(self, subject):
        # > y_true <- c(1,0,1,1,1,0,1,1,0,0,0,1)
        # > y_pred <- c(1,0,1,0,1,1,1,0,0,1,1,1)
        # > var(y_true)
        # [1] 0.2651515
        # > b = 0
        # > for(a in y_true){ b = b + (mean(y_true) - a)^2};b/length(y_true)
        # [1] 0.2430556

        # > var(y_pred)
        # [1] 0.2424242
        # > b = 0
        # > for(a in y_pred){ b = b + (mean(y_pred) - a)^2};b/length(y_pred)
        # [1] 0.2222222
        y_true = [1,0,1,1,1,0,1,1,0,0,0,1]
        y_pred = [1,0,1,0,1,1,1,0,0,1,1,1]
        result = subject.evaluate(y_true, y_pred)

        assert abs(result[0] - 0.2430556) < 0.000001
        assert abs(result[1] - 0.2222222) < 0.000001








