from unittest.mock import Mock

import numpy as np
import pytest

from learner.machine_learning_models.feature_selector import FeatureSelector
from learner.machine_learning_models.grid_search_mine import GridSearchMine
from sklearn.grid_search import BaseSearchCV, ParameterGrid



class TestFeatureSelector:

    @pytest.fixture()
    def subject(self):
        subject = FeatureSelector()
        return subject

    @pytest.fixture()
    def mlmodel(self):
        mlmodel = Mock()
        mlmodel.was_trained = True
        mlmodel.skmodel.coef_ = [[1, 2, 3, 100, 1, 9, 13, 4, 1, 0, -99, 12]]
        mlmodel.x_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
        return mlmodel

    @pytest.fixture()
    def elastic_mlmodel(self, mlmodel):
        mlmodel.skmodel.coef_ = mlmodel.skmodel.coef_[0]
        mlmodel.skmodel.sparse_coef_.indices = list(range(len(mlmodel.x_names)))
        mlmodel.skmodel.sparse_coef_.data = mlmodel.skmodel.coef_
        return mlmodel

    # Init
    def test_subject_is_a_feature_selector(self, subject):
        assert isinstance(subject, FeatureSelector)

    # determine_best_variables
    def test_determine_best_variables_should_do_nothing_if_the_model_was_not_trained(self, subject):
        mlmodel = Mock()
        mlmodel.was_trained = False
        result = subject.determine_best_variables(mlmodel)
        assert result == None

    def test_determine_best_variables_should_assert_that_the_length_of_coefficients_equals_the_xnames(self, subject):
        mlmodel = Mock()
        mlmodel.was_trained = True
        mlmodel.skmodel.coef_ = [[1,2,3,4]]
        mlmodel.x_names = ['a','b','c']

        with pytest.raises(AssertionError) as err:
            result = subject.determine_best_variables(mlmodel)
        assert err != None

    def test_determine_best_variables_should_return_the_top_variables(self, subject, mlmodel):
        result = subject.determine_best_variables(mlmodel, top=3)
        expected = [['d',100],['k',-99],['g',13]]
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize('top', [1,2,3,4,5,6,7,8])
    def test_determine_best_variables_should_use_the_top_param(self, top, subject, mlmodel):
        result = subject.determine_best_variables(mlmodel, top=top)
        assert len(result) == top

    def test_determine_best_variables_should_return_all_params_if_top_is_too_big(self, subject, mlmodel):
        result = subject.determine_best_variables(mlmodel, top=10000)
        assert len(result) == len(mlmodel.x_names)

    # determine_best_variables_elastic_net
    def test_determine_best_variables_elastic_net_should_do_nothing_if_the_model_was_not_trained(self, subject):
        mlmodel = Mock()
        mlmodel.was_trained = False
        result = subject.determine_best_variables_elastic_net(mlmodel)
        assert result == None

    def test_determine_best_variables_elastic_net_should_assert_that_the_length_of_coefficients_equals_the_xnames(self, subject):
        mlmodel = Mock()
        mlmodel.was_trained = True
        mlmodel.skmodel.coef_ = [1,2,3,4]
        mlmodel.x_names = ['a','b','c']

        with pytest.raises(AssertionError) as err:
            result = subject.determine_best_variables_elastic_net(mlmodel)
        assert err != None

    def test_determine_best_variables_elastic_net_should_return_the_top_variables(self, subject, elastic_mlmodel):
        result = subject.determine_best_variables_elastic_net(elastic_mlmodel, top=3)
        expected = [['d',100],['k',-99],['g',13]]
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize('top', [1,2,3,4,5,6,7,8])
    def test_determine_best_variables_elastic_net_should_use_the_top_param(self, top, subject, elastic_mlmodel):
        result = subject.determine_best_variables_elastic_net(elastic_mlmodel, top=top)
        assert len(result) == top

    def test_determine_best_variables_elastic_net_should_return_all_params_if_top_is_too_big(self, subject, elastic_mlmodel):
        result = subject.determine_best_variables_elastic_net(elastic_mlmodel, top=10000)
        assert len(result) == len(elastic_mlmodel.x_names)