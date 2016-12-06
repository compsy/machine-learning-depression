import inspect

import math
import sklearn
from sklearn.grid_search import ParameterGrid
from queue import Queue

from learner.machine_learning_models.distributed_grid_search import DistributedGridSearch
import pytest
import numpy as np
import logging
from mpi4py import MPI
from random import shuffle


class TestDistributedGridSearch():
    @pytest.fixture()
    def subject(self):
        ml_model = 'ml_model'
        estimator = 'estimator'
        param_grid = {'a': [1,2,3,4], 'b': [5,6,7,8], 'c': [4,3,2,1] }
        cv = 123
        subject = DistributedGridSearch(ml_model, estimator, param_grid, cv)
        return subject

    def test_initialize_should_set_the_correct_parameters(self):
        ml_model = 'ml_model'
        estimator = 'estimator'
        param_grid = [123]
        cv = 123
        cpus_per_node = 321

        subject = DistributedGridSearch(ml_model, estimator, param_grid, cv, cpus_per_node)

        assert subject.skmodel == estimator

        # It should create a parameter grid from the param grid
        assert type(subject.param_grid) == ParameterGrid
        assert subject.cv == cv
        assert subject.ml_model == ml_model
        assert subject.cpus_per_node == cpus_per_node

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
        subject = DistributedGridSearch(ml_model, estimator, param_grid, cv, n_iter)

        assert subject.comm == commworld()

    # fit
    def test_fit_should_call_slave_if_rank_is_not_0_and_return_false(self, subject, monkeypatch):
        ranks = list(map(lambda a: a + 1, range(10)))
        X = [[1],[2],[3]]
        y = [3,2,1]
        def fake_slave(fake_X, fake_y):
            assert fake_X == X
            assert fake_y == y
            return 'slave'

        monkeypatch.setattr(subject, 'slave', fake_slave)
        for rank in enumerate(ranks):
            subject.rank = rank
            result = subject.fit(X,y)
            assert result == False

    def test_fit_should_call_slave_if_rank_is_not_0_and_return_the_master(self, subject, monkeypatch):
        X = [[1], [2], [3]]
        y = [3, 2, 1]

        def fake_master():
            return 'master'

        monkeypatch.setattr(subject, 'master', fake_master)
        subject.rank = 0
        result = subject.fit(X, y)
        assert result == 'master'

    # create_job_queue
    def test_create_job_queue_returns_a_queue(self, subject):
        result = subject.create_job_queue(True)
        assert type(result) == Queue

    def test_create_job_queue_returns_a_queue_with_the_correct_elements(self, subject):
        subject.cpus_per_node = 8
        subject.workers = 10
        result = subject.create_job_queue(True)
        jobs = result.get()
        while jobs != StopIteration:
            job_result = [job < len(subject.param_grid) for job in jobs]
            # It should return a list
            assert type(jobs) == list

            # All elements in the list should be included in the paramgrid
            assert np.all(job_result)
            jobs = result.get()

    def test_create_job_queue_returns_a_queue_that_is_sorted_if_no_shuffle_flag_is_provided(self, subject):
        subject.cpus_per_node = 8
        subject.workers = 10
        result = subject.create_job_queue(False)
        jobs = result.get()
        prev = -1
        while jobs != StopIteration:
            for job in jobs:
                assert job > prev
                prev = job

            jobs = result.get()

    def test_create_job_queue_returns_a_queue_that_is_not_sorted_if_a_shuffle_flag_is_provided(self, subject):
        subject.cpus_per_node = 8
        subject.workers = 10
        result = subject.create_job_queue(True)
        jobs = result.get()
        prev = -1
        res = []
        while jobs != StopIteration:
            for job in jobs:
                res.append(job < prev)
                prev = job

            jobs = result.get()

        # It's almost certain that the resulting arrays are not completely ordered, if they are shuffeled
        assert np.any(res)

    def test_create_job_queue_returns_a_queue_with_jobs_according_to_the_number_of_workers(self, subject):
        result = subject.create_job_queue(True)
        assert result.qsize() == round(len(subject.param_grid) / subject.cpus_per_node, 0)

        subject.cpus_per_node = 8
        subject.workers = 10
        result = subject.create_job_queue(True)
        expected_without_kills =  math.ceil(len(subject.param_grid) / subject.cpus_per_node)

        # A job is added to kill all workers
        expected = expected_without_kills + subject.workers

        assert result.qsize() == expected

        jobs = result.get()
        assert len(jobs) == subject.cpus_per_node

    def test_create_job_queue_should_add_kill_jobs_at_the_end(self, subject):
        subject.cpus_per_node = 8
        subject.workers = 10
        result = subject.create_job_queue(True)
        expected_without_kills = math.ceil(len(subject.param_grid) / subject.cpus_per_node)

        # all last jobs should be kill commands
        while expected_without_kills > 0:
            result.get()
            expected_without_kills -= 1

        while result.qsize() > 0:
            job = result.get()
            assert job == StopIteration


    # master
    @pytest.mark.skip(reason="no way of currently testing this")
    def test_master(self):
        assert False

    # slave
    @pytest.mark.skip(reason="no way of currently testing this")
    def test_slave(self):
        assert False