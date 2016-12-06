import numpy as np
import pytest
from learner.machine_learning_models.grid_search_mine import GridSearchMine
from sklearn.grid_search import BaseSearchCV, ParameterGrid



class TestGridSearchMine:

    @pytest.fixture()
    def subject(self):
        estimator = 'estimator'
        param_grid = ParameterGrid({'a': [123,2], 'b': [4,5]})
        subject = GridSearchMine(estimator, param_grid)
        return subject

    # Init
    def test_grid_search_mine_is_subclass_of_base_search_cv(self, subject):
        assert isinstance(subject, BaseSearchCV)

    def test_init_should_just_set_the_param_grid(self):
        estimator = 'estimator'
        param_grid = ParameterGrid({'a': [123, 2], 'b': [4, 5]})
        subject = GridSearchMine(estimator, param_grid)
        assert subject.param_grid == param_grid

    # fit
    def test_fit_should_call_the_inherited_fit_method(self, subject, monkeypatch):
        X = [[1,2,3],[4,5,6],[7,8,9]]
        y = [4,5,6]
        param_grid = subject.param_grid
        def fake_fit(fake_X, fake_y, fake_param_grid):
            assert fake_X == X
            assert fake_y == y
            assert fake_param_grid == fake_param_grid

        monkeypatch.setattr(subject, '_fit', fake_fit)

        subject.fit(X, y)