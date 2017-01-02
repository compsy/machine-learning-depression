
from unittest.mock import MagicMock, Mock
import pytest
import pandas as pd
from sklearn.grid_search import GridSearchCV
from learner.models.participant import Participant
import numpy as np


@pytest.fixture()
def mock_cacher(monkeypatch):
    def fake_write_cache(data, cache_name):
        return None

    mock_cacher = Mock()
    monkeypatch.setattr(mock_cacher, 'write_cache', fake_write_cache)
    return mock_cacher

@pytest.fixture()
def mock_reader():
    file_data = pd.DataFrame([[1, 1.5], [2, 2.5]], columns=['pident', 'avalue'])
    mock_reader = Mock()
    mock_reader.read_file = MagicMock(return_value=file_data)
    return mock_reader

@pytest.fixture()
def mock_skmodel(monkeypatch):
    def fake_get_params(deep=True):
        return {'a': 1}

    file_data = 'running fitting'
    mock_fitted_skmodel = Mock()
    mock_fitted_skmodel.fitted = file_data
    monkeypatch.setattr(mock_fitted_skmodel, 'get_params', fake_get_params)

    def fake_fit(X, y):
        return mock_fitted_skmodel

    mock_skmodel = Mock()
    monkeypatch.setattr(mock_skmodel, 'fit', fake_fit)

    return mock_skmodel

@pytest.fixture()
def mock_gridsearch_skmodel(monkeypatch, mock_skmodel):
    mock_fitted_bestmodel_skmodel = mock_skmodel.fit(1,2)
    mock_fitted_bestmodel_skmodel.fitted = 'best_estimator!'

    mock_fitted_gridsearch_skmodel = GridSearchCV('estimator', {'a': [1, 2], 'b': [3, 4]})
    mock_gridsearch_skmodel = GridSearchCV('estimator', {'a':[1,2],'b':[3,4]})

    def fake_get_params(deep=True):
        return {'a': 1}

    monkeypatch.setattr(mock_fitted_gridsearch_skmodel, 'get_params', fake_get_params)

    file_data = 'running fitting'
    mock_fitted_gridsearch_skmodel.fitted = file_data
    mock_fitted_gridsearch_skmodel.best_estimator_ = mock_fitted_bestmodel_skmodel

    def fake_fit(X, y):
        return mock_fitted_gridsearch_skmodel

    monkeypatch.setattr(mock_gridsearch_skmodel, 'fit', fake_fit)
    return mock_gridsearch_skmodel


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


################# For questionnaires
@pytest.fixture
def mock_get_field(request, subject, monkeypatch):
    my_participant = request.param['participant']
    field_name = request.param['field']
    val = request.param['value']

    def fake_get_field(participant, field):
        assert field_name == field
        assert my_participant == participant
        return val

    monkeypatch.setattr(subject, 'get_field', fake_get_field)
