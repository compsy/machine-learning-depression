import inspect
from learner.machine_learning_evaluation.mse_evaluation import MseEvaluation, RootMseEvaluation
import pytest
import numpy as np

import logging

class TestMseEvaluation():
    @pytest.fixture()
    def subject(self):
        subject = MseEvaluation()
        return subject

    @pytest.fixture()
    def subject_root(self):
        subject = RootMseEvaluation()
        return subject

    def test_mse_initialization(self, subject):
        assert subject.name == 'MSE Evaluator'
        assert subject.model_type == 'regression'

    def test_evaluate_mse_evaluator(self, subject):
        # > y_true <- c(1,0,1,1,1,0,1,1,0,0,0,1)
        # > y_pred <- c(1,0,1,0,1,1,1,0,0,1,0,1)
        # > mse <- sum((y_true - y_pred)^2)/length(y_true)
        # > mse
        # [1] 0.3333333
        y_true = [1,0,1,1,1,0,1,1,0,0,0,1]
        y_pred = [1,0,1,0,1,1,1,0,0,1,0,1]
        result = subject.evaluate(y_true, y_pred)
        assert result == 1/3

    def test_rmse_initialization(self, subject_root):
        assert subject_root.name == 'Root MSE Evaluator'
        assert subject_root.model_type == 'regression'

    def test_evaluate_root_mse_evaluator(self, subject_root):
        # > y_true <- c(1,0,1,1,1,0,1,1,0,0,0,1)
        # > y_pred <- c(1,0,1,0,1,1,1,0,0,1,0,1)
        # > mse <- sum((y_true - y_pred)^2)/length(y_true)
        # > rmse <- sqrt(mse)
        # > rmse
        # [1] 0.5773503
        y_true = [1,0,1,1,1,0,1,1,0,0,0,1]
        y_pred = [1,0,1,0,1,1,1,0,0,1,0,1]
        result = subject_root.evaluate(y_true, y_pred)
        assert abs(result -  0.5773503) < 0.0000001



