from learner.models import questionnaire, participant
from learner.data_input import spss_reader
from unittest.mock import MagicMock
import pandas as pd
import pytest



class TestQuestionnaire:
    def test_get_header(self):
        True

    @pytest.fixture()
    def mock_reader(self):
        returnvalue = pd.DataFrame([[1,1.5], [2,2.5]], columns=['pident', 'avalue'])
        mock_reader = spss_reader.SpssReader()
        mock_reader.read_file = MagicMock(return_value=returnvalue)
        return mock_reader

    @pytest.fixture()
    def mock_participant(self):
        mock_participant = participant.Participant(pident=1, sexe=1, age=1)
        return mock_participant

    @pytest.fixture()
    def subject(self, mock_reader):
        function_mapping = {'value': 2}
        subject = questionnaire.Questionnaire(name='name',
                                              filename='name.csv',
                                              measurement_moment='a',
                                              reader=mock_reader,
                                              function_mapping=function_mapping)
        return subject

    def test_get_field(self, subject, mock_participant):
        result = subject.get_field(mock_participant, 'value')
        assert result == 1.5

    def test_get_header(self, subject):
        # TODO: Possible bug, the dict sorts the keys
        result = list(subject.get_header())
        assert len(result) == 1
        assert result[0] == 'aname-value'


    def test_variable_name(self, subject):
        variable_name = 'test'
        result = subject.variable_name(variable_name)
        expected = 'a' + variable_name
        assert result == expected

    def test_number_of_variables(self, subject):
        expected = 1
        result = subject.number_of_variables()
        assert result == expected

    def test_get_row(self, subject, mock_participant):
        expected  = [1, 1.5]
        result = subject.get_row(mock_participant)
        assert result['pident'] == expected[0]
        assert result['avalue'] == expected[1]

    def test_som_score(self, subject, mock_participant):
        with pytest.raises(NotImplementedError):
            subject.som_score(mock_participant)
