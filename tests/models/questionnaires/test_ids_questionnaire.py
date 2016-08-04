import inspect

from learner.models.questionnaires.ids_questionnaire import IDSQuestionnaire
import pytest
import numpy as np


class TestIDSQuestionnaire:
    name = 'ids'
    filename = 'ids.csv'
    measurement_moment = 'a'

    @pytest.fixture()
    def subject(self, mock_reader):
        subject = IDSQuestionnaire(name=self.name,
                                   filename=self.filename,
                                   measurement_moment=self.measurement_moment,
                                   reader=mock_reader)
        return subject

    def test_init(self, subject):
        # Test whether the correct variables for a some score are set
        expected = [
            'ids01', 'ids02', 'ids03', 'ids04', 'ids05', 'ids06', 'ids07', 'ids08', 'ids09A', 'ids09B', 'ids09C',
            'ids10', 'ids11', 'ids12', 'ids13', 'ids14', 'ids15', 'ids16', 'ids17', 'ids18', 'ids19', 'ids20', 'ids21',
            'ids22', 'ids23', 'ids24', 'ids25', 'ids26', 'ids27', 'ids28'
        ]
        result = subject.variables_for_som_score
        assert np.array_equal(result, expected)

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
            # If the current instance is a string, it will be captured by another function, and won't be evaluated as
            # a function.
            if (isinstance(current, str)): continue
            assert current in all_functions

    def test_som_score_sums_scores(self, subject, monkeypatch, mock_participant):
        fake_data = {}
        index = 0
        total = 0
        for key in subject.variables_for_som_score:
            fake_data[subject.variable_name(key, force_lower_case=False)] = index
            total += index - 1
            index += 1

        # Add one, as the first index = 0 == -1, which is neglected
        total = total + 1

        def fake_get_row(participant):
            return fake_data

        monkeypatch.setattr(subject, 'get_row', fake_get_row)
        result = subject.som_score(mock_participant)
        print(result)
        assert result == total

    def test_som_score_returns_nan(self, subject, monkeypatch, mock_participant):
        fake_data = {}
        for key in subject.variables_for_som_score:
            fake_data[subject.variable_name(key, force_lower_case=False)] = -1

        def fake_get_row(participant):
            return fake_data

        monkeypatch.setattr(subject, 'get_row', fake_get_row)
        result = subject.som_score(mock_participant)
        assert np.isnan(result)

    def test_severity(self, subject, monkeypatch):
        # Used scores from:
        # http://www.ids-qids.org/Severity_Thresholds.pdf

        # Return nan if score is nan
        monkeypatch.setattr(subject, 'som_score', lambda _unused: np.nan)
        assert np.isnan(subject.severity(None))

        # Return 0 if score is < 13
        for i in range(13):
            monkeypatch.setattr(subject, 'som_score', lambda _unused: i)
            assert subject.severity(None) == 0

        # Return 1 if score > 14 < 25
        for i in range(14, 25):
            monkeypatch.setattr(subject, 'som_score', lambda _unused: i)
            assert subject.severity(None) == 1

        # Return 2 if score > 26 < 38
        for i in range(26, 38):
            monkeypatch.setattr(subject, 'som_score', lambda _unused: i)
            assert subject.severity(None) == 2

        # Return 3 if score > 39 < 48
        for i in range(39, 48):
            monkeypatch.setattr(subject, 'som_score', lambda _unused: i)
            assert subject.severity(None) == 3

        # Return 1 if score > 26 < 38
        for i in range(49, 100):
            monkeypatch.setattr(subject, 'som_score', lambda _unused: i)
            assert subject.severity(None) == 4
