import pytest
from learner.machine_learning_evaluation.true_false_ratio_evaluation import TrueFalseRationEvaluation


class TestTrueFalseRatioEvaluation():
    @pytest.fixture()
    def subject(self):
        subject = TrueFalseRationEvaluation(pos_label=1)
        return subject

    def test_mse_initialization(self, subject):
        assert subject.name == 'True False evaluation'
        assert subject.model_type == 'classification'

    def test_evaluate_true_false_evaluator(self, subject):
        # > y_train < - c(1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1)
        # > y_test < - c(1, 0, 1)
        # > length(y_train[y_train == 1])
        # [1]
        # 7
        # length(y_test[y_test == 1])
        #
        # (length(y_train[y_train == 1]) / length(y_train)) * 100
        # (length(y_test[y_test == 1]) / length(y_test)) * 100 > length(y_test[y_test == 1])
        # [1]
        # 2
        #
        # (length(y_train[y_train == 1]) / length(y_train)) * 100
        # (length(y_test[y_test == 1]) / length(y_test)) * 100 >
        # > (length(y_train[y_train == 1]) / length(y_train)) * 100
        # [1]
        # 58.33333
        # (length(y_test[y_test == 1]) / length(y_test)) * 100 > (length(y_test[y_test == 1]) / length(y_test)) * 100
        # [1]
        # 66.66667
        y_train = [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1]
        y_test = [1,0,1]
        ytrain_all, ytrain_perc, ytest_all, ytest_perc = subject.evaluate(y_train, y_test)

        assert ytrain_all == 7
        assert ytrain_perc == 58 + 1/3
        assert ytest_all == 2
        assert ytest_perc == 2/3 * 100

    def test_evaluate_true_false_evaluator_with_other_pos_label(self):
        # > y_train < - c(1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1)
        # > y_test < - c(1, 0, 1)
        # > length(y_train[y_train == 0])
        # [1]
        # 5
        # length(y_test[y_test == 0])
        #
        # (length(y_train[y_train == 0]) / length(y_train)) * 100
        # (length(y_test[y_test == 0]) / length(y_test)) * 100 > length(y_test[y_test == 0])
        # [1]
        # 1
        #
        # (length(y_train[y_train == 0]) / length(y_train)) * 100
        # (length(y_test[y_test == 0]) / length(y_test)) * 100 >
        # > (length(y_train[y_train == 0]) / length(y_train)) * 100
        # [1]
        # 41.66667
        # (length(y_test[y_test == 0]) / length(y_test)) * 100 > (length(y_test[y_test == 0]) / length(y_test)) * 100
        # [1]
        # 33.33333
        subject = TrueFalseRationEvaluation(pos_label=0)
        y_train = [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1]
        y_test = [1, 0, 1]
        ytrain_all, ytrain_perc, ytest_all, ytest_perc = subject.evaluate(y_train, y_test)

        assert ytrain_all == 5
        assert abs(ytrain_perc - (41 + 2 / 3)) <= 1e-12

        assert ytest_all == 1
        assert ytest_perc == 1 / 3 * 100