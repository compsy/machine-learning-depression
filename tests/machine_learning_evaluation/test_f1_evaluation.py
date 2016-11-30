import inspect
from learner.machine_learning_evaluation.f1_evaluation import F1Evaluation
import pytest
import numpy as np

import logging

class TestF1Evaluation():
    @pytest.fixture()
    def subject(self):
        subject = F1Evaluation()
        return subject

    @pytest.fixture()
    def subject_true_is_1(self):
        subject = F1Evaluation(pos_label=1)
        return subject

    def test_initialization(self, subject):
        assert subject.name == 'F1 Evaluator'
        assert subject.model_type == 'classification'

    def test_evaluate_with_zero_as_true_class(self, subject):
        # > true_data <- c(0,1,0,0,0,1,0,0,1,1,1,0)
        # > pred_data <- c(0,1,0,1,0,0,0,1,1,0,1,0)
        # > tn = (true_data & pred_data) * 1
        # > tn = (!true_data & !pred_data) * 1
        # > fp = (!true_data & pred_data) * 1
        # > fn = (true_data & !pred_data) * 1
        # > precision = sum(tp) / (sum(tp) + sum(fp))
        # > recall = sum(tp) / (sum(tp) + sum(fn))
        # > F1 = 2 * ((precision * recall) / (precision + recall))
        # > F1
        # [1] 0.6
        y_true = [1,0,1,1,1,0,1,1,0,0,0,1]
        y_pred = [1,0,1,0,1,1,1,0,0,1,0,1]
        result = subject.evaluate(y_true, y_pred)
        assert result == 0.6

    def test_evaluate_with_one_as_true_class(self, subject_true_is_1):
        # > true_data <- c(1,0,1,1,1,0,1,1,0,0,0,1)
        # > pred_data <- c(1,0,1,0,1,1,1,0,0,1,0,1)
        # > tp = (true_data & pred_data) * 1
        # > tn = (!true_data & !pred_data) * 1
        # > fp = (!true_data & pred_data) * 1
        # > fn = (true_data & !pred_data) * 1
        # > precision = sum(tp) / (sum(tp) + sum(fp))
        # > recall = sum(tp) / (sum(tp) + sum(fn))
        # > F1 = 2 * ((precision * recall) / (precision + recall))
        # > F1
        # [1] 0.7142857
        y_true = [1,0,1,1,1,0,1,1,0,0,0,1]
        y_pred = [1,0,1,0,1,1,1,0,0,1,0,1]
        result = subject_true_is_1.evaluate(y_true, y_pred)
        assert abs(result -  0.7142857) < 0.000001


