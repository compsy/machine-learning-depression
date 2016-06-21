import inspect

from learner.models.questionnaires.bai_questionnaire import BAIQuestionnaire
import pytest
import numpy as np


class TestBAIQuestionnaire:
    name = 'baiQuestionnaire'
    filename = 'bai.csv'
    measurement_moment = 'e'

    @pytest.fixture()
    def subject(self, mock_reader):
        subject = BAIQuestionnaire(name=self.name,
                                   filename=self.filename,
                                   measurement_moment=self.measurement_moment,
                                   reader=mock_reader)
        return subject

    @pytest.fixture
    def mock_get_field(self, request, subject, monkeypatch):
        my_participant = request.param['participant']
        field_name = request.param['field']
        val = request.param['value']

        def fake_get_field(participant, field):
            assert field_name == field
            assert my_participant == participant
            return val

        monkeypatch.setattr(subject, 'get_field', fake_get_field)

    def test_init(self, subject):
        # Test if the super class is called with the correct parameters
        assert subject.name == self.name
        assert subject.filename == self.filename
        assert subject.measurement_moment == self.measurement_moment

    def test_correct_function_mapping(self, subject):
        result = subject.function_mapping
        assert result is not None

        # Test if all exported functions are defined
        all_functions = inspect.getmembers(subject, predicate=inspect.ismethod)
        all_functions = list(map(lambda function: function[1], all_functions))
        for funckey in result.keys():
            current = result[funckey]
            assert current in all_functions

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'baiscal',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'baiscal',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'baiscal',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_total_score(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.total_score('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'baisub',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'baisub',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'baisub',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_subjective_scale_score(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.subjective_scale_score('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'baisev',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'baisev',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'baisev',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_severity_score(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.severity_score('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'baisom',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'baisom',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'baisom',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_somatic_scale_score(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.somatic_scale_score('participant'), expected)
