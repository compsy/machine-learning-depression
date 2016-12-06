import inspect
from learner.machine_learning_evaluation.explained_variance_evaluation import ExplainedVarianceEvaluation
import pytest
import numpy as np

import logging


class TestExplainedVarianceEvaluation():
    @pytest.fixture()
    def subject(self):
        subject = ExplainedVarianceEvaluation()
        return subject

    def test_initialization(self, subject):
        assert subject.name == 'Explained Variance Evaluator'
        assert subject.model_type == 'regression'

    def test_evaluate(self, subject):
        #> a = c(1,2,3,4,5,6,7,8,9)
        # > b = c(1.1,1.9,3,4.1,5.2,5.7,7,8,9)
        # > lm(a~b)

        # Call:
        # lm(formula = a ~ b)

        # Coefficients:
        # (Intercept)            b
        # -0.02874      1.00575

        # > c = lm(a~b)
        # > summary(c)

        # Call:
        # lm(formula = a ~ b)

        # Residuals:
        # Min       1Q   Median       3Q      Max
        # -0.20115 -0.07759 -0.01724  0.01149  0.29598

        # Coefficients:
        # Estimate Std. Error t value Pr(>|t|)
        # (Intercept) -0.02874    0.10977  -0.262    0.801
        # b            1.00575    0.01954  51.483 2.73e-10 ***
        # ---
        # Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

        # Residual standard error: 0.1503 on 7 degrees of freedom
        # Multiple R-squared:  0.9974,    Adjusted R-squared:  0.997
        # F-statistic:  2650 on 1 and 7 DF,  p-value: 2.733e-10

        y_true = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        y_pred = [1.1, 1.9, 3, 4.1, 5.2, 5.7, 7, 8, 9]
        result = subject.evaluate(y_true, y_pred)
        assert abs(result -  0.9974) < 0.0001

