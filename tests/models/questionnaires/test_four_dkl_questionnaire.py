import inspect

from learner.models.questionnaires.four_dkl_questionnaire import FourDKLQuestionnaire
import pytest
import numpy as np


class TestFourDKLQuestionnaire:
    name = '4dkl'
    filename = '4dkl.csv'
    measurement_moment = 'a'

    @pytest.fixture()
    def subject(self, mock_reader):
        subject = FourDKLQuestionnaire(name=self.name,
                                       filename=self.filename,
                                       measurement_moment=self.measurement_moment,
                                       reader=mock_reader)
        return subject

    def test_init(self, subject):
        # Test whether the correct variables for a some score are set
        expected = [
            '4dkld01', '4dkld02', '4dkld03', '4dkld04', '4dkld05', '4dkld06', '4dkld07', '4dkld08', '4dkld09',
            '4dkld10', '4dkld11', '4dkld12', '4dkld13', '4dkld14', '4dkld15', '4dkld16'
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
            assert current in all_functions

    def test_som_score_sums_scores(self, subject, monkeypatch, mock_participant):
        fake_data = {}
        index = 0
        total = 0
        for key in subject.variables_for_som_score:
            fake_data[subject.variable_name(key)] = index
            total += index
            index += 1

        def fake_get_row(participant):
            return fake_data

        monkeypatch.setattr(subject, 'get_row', fake_get_row)
        result = subject.som_score(mock_participant)
        print(result)
        assert result == total

    def test_som_score_returns_nan(self, subject, monkeypatch, mock_participant):
        fake_data = {}
        for key in subject.variables_for_som_score:
            fake_data[subject.variable_name(key)] = 0

        def fake_get_row(participant):
            return fake_data

        monkeypatch.setattr(subject, 'get_row', fake_get_row)
        result = subject.som_score(mock_participant)
        assert np.isnan(result)


