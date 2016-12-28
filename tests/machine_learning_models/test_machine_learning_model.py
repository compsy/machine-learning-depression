import numpy as np
from unittest.mock import MagicMock, Mock
from numpy import logspace
from scipy.stats import expon, halflogistic
import pytest
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from learner.machine_learning_models.machine_learning_model import MachineLearningModel
from learner.machine_learning_models.distributed_grid_search import DistributedGridSearch
from learner.machine_learning_models.randomized_search_mine import RandomizedSearchMine
from learner.machine_learning_models.distributed_random_grid_search import DistributedRandomGridSearch


class TestMachineLearningModel:
    @pytest.fixture()
    def subject(self, mock_cacher):
        x = np.array([[1], [ 2 ], [ 3 ], [ 4 ], [ 5 ], [ 6 ], [ 7 ], [ 8 ], [ 9 ], [ 10 ]])
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])[::-1]
        x_names = np.array(['input'])
        y_names = np.array(['output'])

        subject = MachineLearningModel(x, y, x_names, y_names, hyperparameters=None)
        subject.cacher = mock_cacher
        return subject

    # init
    def test_init_should_divide_the_x_and_y_correctly(self, subject):
        assert np.all(subject.x >= 1)
        assert np.all(subject.y < 1)

    # remove missings
    def test_remove_missings_should_remove_missings(self, subject):
        data = [[100, 100, 100], [1, 2, 3], [100, 8, np.nan]]
        expected = np.mean([100, 3])
        expected_data = [[100, 100, 100], [1, 2, 3], [100, 8, expected]]
        result = subject.remove_missings(data)
        print(result)
        assert np.array_equal(result, expected_data)

    # train_test_data
    def test_train_test_data_should_split_up_the_data_in_train_and_test_set(
            self, subject):
        result = subject.train_test_data()

        # it should return an x train and y train set, an xtest and ytest set
        assert len(result) == 4

        print(result)

        # It should divide the test / train set according to the test size
        # Train set
        assert len(result[0]) == len(result[2])
        # Test set
        assert len(result[1]) == len(result[3])

        assert len(result[0]) == (1 - subject.test_size) * len(subject.x)
        assert len(result[1]) == subject.test_size * len(subject.x)

        # the data should be correct (all y are < 1 for testing purposes)
        assert np.all(result[0] >= 1)
        assert np.all(result[1] >= 1)
        assert np.all(result[2] < 1)
        assert np.all(result[3] < 1)

        # All data should be included in the default set
        assert len(np.append(result[0], result[1])) == len(subject.x)

        for x_val in np.append(result[0], result[1]):
            assert x_val in subject.x

        assert len(np.append(result[2], result[3])) == len(subject.y)
        for y_val in np.append(result[3], result[3]):
            assert y_val in subject.y

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
            assert np.array_equal(X, subject.x_train)
            assert np.array_equal(y, subject.y_train)
            mock_skmodel.fitting = 'fitting!'
            return mock_skmodel

        monkeypatch.setattr(mock_skmodel, 'fit', fake_fit)
        subject.grid_model = None
        subject.skmodel = mock_skmodel

        result = subject.train()
        assert result.fitting == 'fitting!'

    def test_train_should_call_gridmodel_fit_if_a_gridmodel_is_available(self, subject, monkeypatch, mock_skmodel):
        def fake_gridmodel_fit(X, y):
            assert np.array_equal(X, subject.x_train)
            assert np.array_equal(y, subject.y_train)
            mock_skmodel.fitting = 'fitting! gridmodel'
            return mock_skmodel

        monkeypatch.setattr(mock_skmodel, 'fit', fake_gridmodel_fit)
        subject.grid_model = mock_skmodel
        subject.skmodel = mock_skmodel
        result = subject.train()
        assert result.fitting == 'fitting! gridmodel'

    def test_train_should_only_update_the_result_if_it_came_from_the_master_node(self, subject, monkeypatch, mock_skmodel):
        def fake_fit(X, y):
            assert np.array_equal(X, subject.x_train)
            assert np.array_equal(y, subject.y_train)
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
            assert np.array_equal(x, subject.x_train)
            assert np.array_equal(y, subject.y_train)
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

        subject.grid_model = mock_gridsearch_skmodel
        subject.skmodel = mock_gridsearch_skmodel
        subject.train()
        assert subject.skmodel.fitted == return_val

    def test_train_should_cache_the_correct_file_and_data(self, subject, monkeypatch, mock_skmodel):
        def fake_write_cache(data, cache_name):
            assert data == {'a': 1}
            assert cache_name == subject.short_name + '_hyperparameters.pkl'
            return None

        monkeypatch.setattr(subject.cacher, 'write_cache', fake_write_cache)
        subject.skmodel = mock_skmodel
        subject.train()


    # scoring
    def test_scoring_should_return_MSE_for_default_models(self, subject):
        subject.model_type = 'models'
        assert subject.scoring() == 'mean_squared_error'

    def test_scoring_should_return_accuracy_for_default_models(self, subject):
        subject.model_type = 'classification'
        assert subject.scoring() == 'accuracy'

    def test_scoring_should_raise_for_non_existing_category(self, subject):
        subject.model_type = 'this_is_fake'
        with pytest.raises(NotImplementedError, message = 'Type: this_is_fake not implemented'):
            subject.scoring()


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
        assert subject.short_name == type(subject).__name__
        assert subject.short_name == 'MachineLearningModel'

    # grid_search
    def test_grid_search_should_consider_the_hpc_parameter(self, subject):
        grid = [{'kernel': ['sigmoid'],
                        'C': [1, 10, 100, 1000],
                        'coef0': logspace(0, 1, 5),
                        'gamma': logspace(0, 1, 5)}]

        random_grid = [{'kernel': ['sigmoid'],
                           'C': halflogistic(scale=100),
                           'gamma': halflogistic(scale=.1),
                           'coef0': halflogistic(scale=.1)}]

        subject.grid_search_type = 'exhaustive'

        subject.hpc = True
        result = subject.grid_search(grid, random_grid)
        assert isinstance(result, DistributedGridSearch)

        subject.hpc = False
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


        subject.hpc = True
        subject.grid_search_type = 'exhaustive'
        result = subject.grid_search(grid, random_grid)
        assert isinstance(result, DistributedGridSearch)

        subject.grid_search_type = 'random'
        result = subject.grid_search(grid, random_grid)
        assert isinstance(result, DistributedRandomGridSearch)

        subject.hpc = False
        subject.grid_search_type = 'exhaustive'
        result = subject.grid_search(grid, random_grid)
        assert isinstance(result, GridSearchCV)

        subject.grid_search_type = 'random'
        result = subject.grid_search(grid, random_grid)
        assert isinstance(result, RandomizedSearchMine)

    # predict_for_roc
    def test_predict_for_roc(self, monkeypatch, subject, mock_skmodel):
        x_data = [1,2,3]
        def fake_predict_proba(fake_x_data):
            assert fake_x_data == x_data
            return np.array([['faked', 'fake']])

        monkeypatch.setattr(mock_skmodel, 'predict_proba', fake_predict_proba)
        subject.skmodel = mock_skmodel

        result = subject.predict_for_roc(x_data)
        assert result == np.array(['fake'])

