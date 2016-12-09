import numpy as np
import pandas
from learner.data_input.spss_reader import SpssReader
from rpy2.robjects import pandas2ri
import pytest


class TestSpssReader:
    @pytest.fixture()
    def subject(self):
        subject = SpssReader()
        return subject

    def test_read_file_returns_a_pandasdataframe(self, subject):
        # monkeypatch.setattr(subject, 'get_row', fake_get_row)
        result = subject.read_file('N1_A259D.sav')
        assert type(result) is pandas.core.frame.DataFrame

    def test_it_reads_the_file_provided_to_it(self, subject, monkeypatch):
        fake_base_dir = 'testtest/'
        expected = 'test'

        def fake_read_spss(file_name, **kwargs):
            assert file_name == fake_base_dir + expected
            raise ValueError('stop_execution')

        monkeypatch.setattr(subject, 'read_spss', fake_read_spss)
        monkeypatch.setattr(subject, 'base_dir', fake_base_dir)
        with pytest.raises(ValueError) as err:
            result = subject.read_file('test')
        assert str(err.value) == 'stop_execution'

    def test_returns_the_correct_data(self, subject, monkeypatch):
        fake_base_dir = 'tests/data_examples/'
        monkeypatch.setattr(subject, 'base_dir', fake_base_dir)
        result = subject.read_file('test_data.sav')
        expected_columns = [
            'pident', 'Sexe', 'Age', 'aframe01', 'aframe02', 'aarea',
            'aeducat', 'aeducasp', 'aedu', 'aedulvl', 'abthctry', 'abcspec',
            'anatnmbr', 'anation1', 'anation2', 'anatspec', 'anorthea'
        ]
        expected_data = [1000001, 2, 49, 3, 3, 1, 7, '', 15, 3, 1 ,'', 1, 1, -2, '', 1
        ]

        assert result.empty is False
        assert np.array_equal(result.shape, (1, len(expected_columns)))
        assert np.array_equal(expected_columns, result.columns)

        for index, entry in result.iterrows():
            current = entry[expected_columns[index]]
            assert np.array_equal(current, expected_data[index])

