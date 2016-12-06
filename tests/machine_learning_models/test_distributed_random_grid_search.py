import inspect
from learner.machine_learning_models.distributed_random_grid_search import DistributedRandomGridSearch
import pytest
import numpy as np
import logging
from mpi4py import MPI
from random import shuffle


class TestDistributedRandomGridSearch():
    @pytest.fixture()
    def subject(self):
        ml_model = 'ml_model'
        estimator = 'estimator'
        param_grid = [123]
        cv = 123
        n_iter = 321
        subject = DistributedRandomGridSearch(ml_model, estimator, param_grid, cv, n_iter)
        return subject

    def test_initialize_should_set_the_correct_parameters(self):
        ml_model = 'ml_model'
        estimator = 'estimator'
        param_grid = [123]
        cv = 123
        n_iter = 321
        subject = DistributedRandomGridSearch(ml_model, estimator, param_grid, cv, n_iter)

        assert subject.skmodel == estimator
        assert subject.param_grid == param_grid
        assert subject.cv == cv
        assert subject.ml_model == ml_model
        assert subject.iterations == n_iter

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_initialize_should_set_the_correct_default_parameters(self, monkeypatch, subject):
        def commworld():
            return 'test'

        monkeypatch.setattr(MPI, 'COMM_WORLD', commworld)

        ml_model = 'ml_model'
        estimator = 'estimator'
        param_grid = [123]
        cv = 123
        n_iter = 321
        subject = DistributedRandomGridSearch(ml_model, estimator, param_grid, cv, n_iter)

        assert subject.comm == commworld()

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_fit(self, monkeypatch, subject):
        assert True

    # get_best_model
    def test_get_best_model_returns_the_best_model(self, monkeypatch, subject):
        scores = range(10)
        models = range(10)
        models = map(lambda a: a * -1, models)
        models = list(zip(scores, models))
        shuffle(models)

        result = subject.get_best_model(models)
        assert type(result) == tuple
        assert np.array_equal(result, (9, -9))

    def test_get_best_model_should_be_able_to_deal_with_none_values(self, monkeypatch, subject):
        scores = range(10)
        models = range(10)
        models = list(map(lambda a: a * -1, models))
        models = list(zip(scores, models))
        models.append(None)

        shuffle(models)

        result = subject.get_best_model(models)
        assert type(result) == tuple
        assert np.array_equal(result, (9, -9))

    def test_get_best_model_should_be_able_to_deal_floats_as_scores(self, monkeypatch, subject):
        scores = range(10)

        scores = list(map(lambda a: a / 10, scores))
        print(scores)

        models = range(10)
        models = list(map(lambda a: - 0.1 * a, models))
        models = list(zip(scores, models))
        models.append(None)

        shuffle(models)

        result = subject.get_best_model(models)
        assert type(result) == tuple
        assert np.array_equal(result, (0.9, -0.9))
