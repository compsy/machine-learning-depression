
from unittest.mock import MagicMock, Mock
import pytest
import pandas as pd
from sklearn.grid_search import GridSearchCV
from learner.models.participant import Participant
import numpy as np


@pytest.fixture()
def mock_reader():
    file_data = pd.DataFrame([[1, 1.5], [2, 2.5]], columns=['pident', 'avalue'])
    mock_reader = Mock()
    mock_reader.read_file = MagicMock(return_value=file_data)
    return mock_reader

@pytest.fixture()
def mock_skmodel():
    file_data = 'running fitting'
    mock_skmodel = Mock()
    mock_skmodel.fit = MagicMock(return_value=file_data)
    return mock_skmodel

@pytest.fixture()
def mock_gridsearch_skmodel():
    mock_skmodel = GridSearchCV('estimator', {'a':[1,2],'b':[3,4]})
    mock_skmodel.fit = MagicMock(return_value=mock_skmodel)
    return mock_skmodel


@pytest.fixture()
def mock_participant():
    mock_participant = Participant(pident=1, sexe=1, age=1)
    return mock_participant

@pytest.fixture()
def expected(request):
    return request.param


@pytest.fixture()
def assert_with_nan():
    def assertion(result, expected):
        if (np.isnan(expected)):
            assert np.isnan(result)
        elif expected is None:
            assert result is expected
        else:
            assert result == expected

    return assertion
