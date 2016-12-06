import numpy as np
import pandas
import inspect
import warnings
from learner.data_transformers.data_transformer import DataTransformer
from rpy2.robjects import pandas2ri
import pytest


class TestDataTransformer():
    @pytest.fixture()
    def subject(self):
        subject = DataTransformer()
        return subject

    def test_get_variable_indices_returns_the_correct_indices(self, subject):
        all_names = np.array(['test1','test2','test3'])

        selected_names = ['test1']
        result = subject.get_variable_indices(all_names, selected_names)
        assert np.array_equal(result, [0])

        selected_names = ['test1', 'test2']
        result = subject.get_variable_indices(all_names, selected_names)
        assert np.array_equal(result, [0,1])

        selected_names = ['test1', 'test2', 'test3']
        result = subject.get_variable_indices(all_names, selected_names)
        assert np.array_equal(result, [0,1,2])

    def test_get_variable_indicises_can_handle_non_existing_keys(self, subject):
        all_names = np.array(['test1','test2','test3'])
        selected_names = ['test81']
        result = subject.get_variable_indices(all_names, selected_names)
        assert np.array_equal(result, [])

        selected_names = ['test81', 'test1']
        result = subject.get_variable_indices(all_names, selected_names)
        assert np.array_equal(result, [0])

    def test_get_variable_index_should_warn_if_not_an_nparray(self,monkeypatch, subject):
        all_names = ['test1','test2','test3']
        def fake_warnings(msg):
            assert msg == 'Note that this function only works on NP arrays!'
            raise ValueError('stop_execution')

        monkeypatch.setattr(warnings, 'warn', fake_warnings)
        with pytest.raises(ValueError) as err:
            subject.get_variable_index(all_names, 'test1')
        assert str(err.value) == 'stop_execution'

    def test_get_variable_index_should_warn_if_a_variable_is_not_found(self, subject, monkeypatch):
        all_names = np.array(['test1','test2','test3'])
        def fake_warnings(msg):
            assert msg == 'Variable not found test81'
            raise ValueError('stop_execution')

        monkeypatch.setattr(warnings, 'warn', fake_warnings)
        with pytest.raises(ValueError) as err:
            subject.get_variable_index(all_names, 'test81')
        assert str(err.value) == 'stop_execution'

    def test_get_variable_index_should_return_none_if_variable_not_found(self, subject):
        all_names = np.array(['test1','test2','test3'])
        result = subject.get_variable_index(all_names, 'test81')
        assert result is None

    def test_get_variable_index_should_return_the_correct_id(self, subject):
        all_names = np.array(['test1','test2','test3'])
        result = subject.get_variable_index(all_names, 'test3')
        assert result == 2
