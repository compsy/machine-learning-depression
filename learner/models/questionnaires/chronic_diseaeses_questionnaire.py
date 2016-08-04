from ..questionnaire import Questionnaire
import numpy as np


class ChronicDiseasesQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):

        #-3 = no scale, too many missings
        # -2 = Qnair not returned
        # 0 = a valid score.
        function_mapping = {
            'numberOfChronicDiseases': self.number_of_chronic_diseases,
            'numberOfChronicDiseasesUnderTreatment': self.number_of_chronic_diseases_under_treatment
        }
        other_available_variables = []
        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

    def number_of_chronic_diseases(self, participant):
        val = self.get_field(participant, 'numdisease')
        return val if val is not None and val >= 0 else np.nan

    def number_of_chronic_diseases_under_treatment(self, participant):
        val = self.get_field(participant, 'numdis_treat')
        return val if val is not None and val >= 0 else np.nan
