import inspect
from learner.factories.questionnaire_factory import QuestionnaireFactory
from learner.models.questionnaire import Questionnaire
import pytest
import numpy as np

import logging

class TestQuestionnaireFactory():
    def test_construct_questionnaires_should_return_a_list_of_questionnaires(self, mock_reader):
        questionnaires = QuestionnaireFactory.construct_questionnaires(mock_reader)
        assert type(questionnaires) is list
        result = [isinstance(questionnaire, Questionnaire) for questionnaire in questionnaires]
        assert np.all(result)

    def test_construct_x_names_should_return_a_list_of_variables(self):
        names = QuestionnaireFactory.construct_x_names()

        assert isinstance(names, np.ndarray)
        result = [isinstance(name, str) for name in names]
        assert np.all(result)

    def test_construct_x_names_should_have_valid_variable_names(self):
        names = QuestionnaireFactory.construct_x_names()
        result = ['-' in name for name in names]
        assert np.all(result)


