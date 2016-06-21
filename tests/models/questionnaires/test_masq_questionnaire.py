import inspect

from learner.models.questionnaires.masq_questionnaire import MASQQuestionnaire
import pytest
import numpy as np

class TestFourDKLQuestionnaire:
    name = '4dkl'
    filename = '4dkl.csv'
    measurement_moment = 'a'

    @pytest.fixture()
    def subject(self, mock_reader):
        subject = MASQQuestionnaire(name= self.name,
                                  filename= self.filename,
                                  measurement_moment= self.measurement_moment,
                                  reader=mock_reader)
        return subject

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

    def test_positive_affect_score(self, subject, monkeypatch, mock_participant):
        def fake_get_field(participant, field_name):
            assert field_name == 'masqpa'
            assert participant == mock_participant
            return val

        val = 0
        monkeypatch.setattr(subject, 'get_field', fake_get_field)
        result = subject.positive_affect_score(mock_participant)
        assert result == val

        val = 1
        monkeypatch.setattr(subject, 'get_field', fake_get_field)
        result = subject.positive_affect_score(mock_participant)
        assert result == val

        val = -1
        monkeypatch.setattr(subject, 'get_field', fake_get_field)
        result = subject.positive_affect_score(mock_participant)
        assert np.isnan(result)

    def test_negative_affect_score(self, subject, monkeypatch, mock_participant):
        def fake_get_field(participant, field_name):
            assert field_name == 'masqna'
            assert participant == mock_participant
            return val

        val = 0
        monkeypatch.setattr(subject, 'get_field', fake_get_field)
        result = subject.negative_affect_score(mock_participant)
        assert result == val

        val = 1
        monkeypatch.setattr(subject, 'get_field', fake_get_field)
        result = subject.negative_affect_score(mock_participant)
        assert result == val

        val = -1
        monkeypatch.setattr(subject, 'get_field', fake_get_field)
        result = subject.negative_affect_score(mock_participant)
        assert np.isnan(result)

    def test_somatization_score(self, subject, monkeypatch, mock_participant):
        def fake_get_field(participant, field_name):
            assert field_name == 'masqsa'
            assert participant == mock_participant
            return val

        val = 0
        monkeypatch.setattr(subject, 'get_field', fake_get_field)
        result = subject.somatization_score(mock_participant)
        assert result == val

        val = 1
        monkeypatch.setattr(subject, 'get_field', fake_get_field)
        result = subject.somatization_score(mock_participant)
        assert result == val

        val = -1
        monkeypatch.setattr(subject, 'get_field', fake_get_field)
        result = subject.somatization_score(mock_participant)
        assert np.isnan(result)