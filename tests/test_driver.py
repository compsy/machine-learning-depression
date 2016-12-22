from unittest.mock import MagicMock, Mock

from learner.models.participant import Participant
from learner.driver import Driver
import pytest
import numpy as np
import pandas as pd


class TestDriver():
    @pytest.fixture()
    def subject(self):
        verbosity = 0
        hpc = False
        polynomial_features = True
        normalize = True
        scale = False
        force_no_caching = True
        feature_selection = True
        subject = Driver(
            verbosity,
            hpc,
            polynomial_features,
            normalize,
            scale,
            force_no_caching,
            feature_selection
        );
        return subject

    # transform_variables
    def test_transform_variables_should_do_nothing_if_all_settings_are_blocked(self, subject):
        x_data = np.array([[1,2,3],[5,6,7],[8,9,10]])
        x_data_orig = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
        x_names = ['a','b','c']

        subject.NORMALIZE = False
        subject.SCALE = False
        subject.POLYNOMIAL_FEATURES = False

        result = subject.transform_variables(x_data, x_names)
        assert np.array_equal(result, x_data_orig)

    def test_transform_variables_normalizes_the_data(self, subject):
        x_data = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
        x_data_orig = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
        x_names = ['a', 'b', 'c']

        subject.NORMALIZE = True
        subject.SCALE = False
        subject.POLYNOMIAL_FEATURES = False

        result = subject.transform_variables(x_data, x_names)

        # Something must have changed
        assert not np.array_equal(result, x_data_orig)
        assert len(result) == len(x_data_orig)
        assert len(result[0]) == len(x_data_orig[0])

        # TODO: Calculate the norm here and compare it


    @pytest.mark.skip(reason="no way of currently testing this")
    def test_transform_variables_scales_the_data(self, subject):
        x_data = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
        x_data_orig = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
        x_names = ['a', 'b', 'c']

        subject.NORMALIZE = False
        subject.SCALE = True
        subject.POLYNOMIAL_FEATURES = False

        result = subject.transform_variables(x_data, x_names)

        # Something must have changed
        assert not np.array_equal(result, x_data_orig)
        assert len(result) == len(x_data_orig)
        assert len(result[0]) == len(x_data_orig[0])

        print(x_data_orig)
        print(result)

        assert False

        # TODO: Calculate the norm here and compare it


    #################################################################################################

    def test_create_participants(self,subject, mock_reader):
        data = [['1', 1, 10], ['2', 2, 50], ['3',1,100]]
        file_data = pd.DataFrame(data, columns=['pident', 'Sexe', 'Age'])
        mock_reader.read_file = MagicMock(return_value=file_data)
        subject.spss_reader = mock_reader

        participants = subject.create_participants()

        assert len(participants) == len(data)
        assert isinstance(participants, dict)
        for index, p in enumerate(participants):
            assert isinstance(participants[p], Participant)
            assert participants[p].pident == int(data[index][0])

            # Gender is converted
            assert participants[p].gender == data[index][1] -1
            assert participants[p].sexe == data[index][1]
            assert participants[p].age == data[index][2]



    def test_calculate_true_false_ratio(self, subject):
        y_data = [0,1,1,1,1,1,0,0,0,1,0,1]
        result = subject.calculate_true_false_ratio(y_data)
        assert len(result) == 2
        assert np.sum(y_data) == result[0]
        assert (np.sum(y_data)/len(y_data)) * 100 == result[1]