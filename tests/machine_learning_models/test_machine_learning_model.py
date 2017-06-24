import numpy as np
import pandas as pd
from unittest.mock import MagicMock, Mock
from numpy import logspace
from scipy.stats import expon, halflogistic
import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from learner.machine_learning_models.machine_learning_model import MachineLearningModel
from learner.machine_learning_models.randomized_search_mine import RandomizedSearchMine


class TestMachineLearningModel:
    @pytest.fixture()
    def subject(self, mock_cacher):
        x_names = np.array(['input'])
        y_names = np.array(['output'])
        x = pd.DataFrame(np.array([[1], [ 2 ], [ 3 ], [ 4 ], [ 5 ], [ 6 ], [ 7 ], [ 8 ], [ 9 ], [ 10 ]]), columns = x_names)
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])[::-1]

        subject = MachineLearningModel(x, y, y_names, hyperparameters=None)
        subject.cacher = mock_cacher
        return subject

    # init
    def test_init_should_divide_the_x_and_y_correctly(self, subject):
        assert np.all(subject.get_x >= 1)
        assert np.all(subject.get_y < 1)

    # print_accuracy

    # print_evaluation

    # train
    def test_train_should_return_true_if_the_model_was_trained(self, subject):
        subject.was_trained = True
        result = subject.train()
        assert result == True

    def test_train_should_raise_not_implemented_error_when_no_subclass(self, subject):
        with pytest.raises(NotImplementedError):
            subject.train()

    def test_train_should_call_skmodel_fit_if_no_gridmodel_is_available(self, subject, monkeypatch, mock_skmodel):

        def fake_fit(X, y):
            assert pd.DataFrame.equals(X, subject.get_x)
            assert np.array_equal(y, subject.get_y)
            mock_skmodel.fitting = 'fitting!'
            return mock_skmodel

        monkeypatch.setattr(mock_skmodel, 'fit', fake_fit)
        subject.grid_model = None
        subject.skmodel = mock_skmodel

        result = subject.train()
        assert result.fitting == 'fitting!'

    def test_train_should_call_gridmodel_fit_if_a_gridmodel_is_available(self, subject, monkeypatch, mock_skmodel):
        def fake_gridmodel_fit(X, y):
            assert pd.DataFrame.equals(X, subject.get_x)
            assert np.array_equal(y, subject.get_y)
            mock_skmodel.fitting = 'fitting! gridmodel'
            return mock_skmodel

        monkeypatch.setattr(mock_skmodel, 'fit', fake_gridmodel_fit)
        subject.grid_model = mock_skmodel
        subject.skmodel = mock_skmodel
        result = subject.train()
        assert result.fitting == 'fitting! gridmodel'

    def test_train_should_only_update_the_result_if_it_came_from_the_master_node(self, subject, monkeypatch, mock_skmodel):
        def fake_fit(X, y):
            assert pd.DataFrame.equals(X, subject.get_x)
            assert np.array_equal(y, subject.get_y)
            # Add an attr so we now this object was returned
            mock_skmodel.fitting = ['fitted', 'data']
            return mock_skmodel

        monkeypatch.setattr(mock_skmodel, 'fit', fake_fit)
        subject.grid_model = None
        subject.skmodel = mock_skmodel

        assert subject.skmodel.fitting != ['fitted', 'data']
        subject.train()
        assert subject.skmodel.fitting == ['fitted', 'data']

        def fake_fit_from_slave(x, y):
            assert pd.DataFrame.equals(X, subject.get_x)
            assert np.array_equal(y, subject.get_y)
            return False

        monkeypatch.setattr(mock_skmodel, 'fit', fake_fit_from_slave)
        subject.grid_model = None
        subject.skmodel = mock_skmodel

        assert subject.skmodel != False
        subject.train()
        assert subject.skmodel != False

    def test_train_should_update_was_trained_if_it_was_trained(self, subject, monkeypatch, mock_skmodel):
        subject.grid_model = None
        subject.skmodel = mock_skmodel

        assert subject.was_trained != True
        subject.train()
        assert subject.was_trained == True

    def test_train_should_get_the_best_estimator_if_the_sk_model_is_a_gridsearchCV(self, monkeypatch, subject, mock_gridsearch_skmodel):
        return_val = 'best_estimator!'

        subject.skmodel = mock_gridsearch_skmodel
        result = subject.train(cache_result=False)
        assert result != True
        assert subject.skmodel.fitted == return_val

    def test_train_should_cache_the_correct_file_and_data(self, subject, monkeypatch, mock_skmodel):
        def fake_write_cache(data, cache_name):
            assert 'score' in data
            assert 'hyperparameters' in data
            assert 'skmodel' in data
            assert 'calculation_time' in data
            assert 'is_bagged' in data

            assert subject.short_name in cache_name
            assert subject.short_name != cache_name + '.pkl'
            return None

        monkeypatch.setattr(subject.cacher, 'write_cache', fake_write_cache)
        subject.skmodel = mock_skmodel
        subject.train()


    # variable_to_validate
    def test_variable_to_validate_should_return_a_correct_string(self,
            subject):
        result = subject.variable_to_validate()
        assert type(result) == str

    # given_name
    def test_given_name_should_return_the_name_of_the_object(self, subject):
        subject.skmodel = MagicMock('skmodel')
        assert subject.given_name == type(subject).__name__ + " Type: " + type(subject.skmodel).__name__
        assert subject.given_name == 'MachineLearningModel Type: MagicMock'

    # short_name
    def test_short_name(self, subject):
        subject.skmodel = MagicMock('skmodel')
        assert subject.short_name == type(subject).__name__
        assert subject.short_name == 'MachineLearningModel'

    def test_short_name_with_bagging(self, subject, monkeypatch):
        subject.skmodel = MagicMock('skmodel')
        monkeypatch.setattr(subject, 'bagged', True)

        assert subject.short_name == type(subject).__name__ + '_bagged'
        assert subject.short_name == 'MachineLearningModel_bagged'

    # grid_search
    def test_grid_search_should_return_the_grid_search_model(self, subject):
        grid = [{'kernel': ['sigmoid'],
                        'C': [1, 10, 100, 1000],
                        'coef0': logspace(0, 1, 5),
                        'gamma': logspace(0, 1, 5)}]

        random_grid = [{'kernel': ['sigmoid'],
                           'C': halflogistic(scale=100),
                           'gamma': halflogistic(scale=.1),
                           'coef0': halflogistic(scale=.1)}]

        subject.grid_search_type = 'exhaustive'

        result = subject.grid_search(grid, random_grid)
        assert isinstance(result, GridSearchCV)



    def test_grid_search_should_take_into_account_grid_search_type(self, subject):
        grid = [{'kernel': ['sigmoid'],
                        'C': [1, 10, 100, 1000],
                        'coef0': logspace(0, 1, 5),
                        'gamma': logspace(0, 1, 5)}]

        random_grid = [{'kernel': ['sigmoid'],
                           'C': halflogistic(scale=100),
                           'gamma': halflogistic(scale=.1),
                           'coef0': halflogistic(scale=.1)}]


        subject.grid_search_type = 'exhaustive'
        result = subject.grid_search(grid, random_grid)
        assert isinstance(result, GridSearchCV)

        subject.grid_search_type = 'random'
        result = subject.grid_search(grid, random_grid)
        assert isinstance(result, RandomizedSearchMine)

    # predict_for_roc
    def test_predict_for_roc(self, monkeypatch, subject, mock_skmodel):
        x_data = pd.DataFrame([1,2,3])
        def fake_predict_proba(fake_x_data):
            assert pd.DataFrame.equals(fake_x_data, x_data)
            return np.array([['faked', 'fake']])

        monkeypatch.setattr(mock_skmodel, 'predict_proba', fake_predict_proba)
        subject.skmodel = mock_skmodel

        result = subject.predict_for_roc(x_data)
        assert result == np.array(['fake'])

