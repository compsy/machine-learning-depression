from ..questionnaire import Questionnaire
import numpy as np


class FourDKLPhysicalComplaintsQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):

        # -1 = too many missings

        function_mapping = {
            'somatizationSumScore': self.somatization_sum_score,
            'somatizationTrychotomization': self.somatization_trychotomization,
            'dichotomizationThrychotomization': self.dichotomization_thrychotomization
        }
        other_available_variables = []
        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

    def somatization_sum_score(self, participant):
        val = self.get_field(participant, '4dsqssc')
        return val if val is not None and val >= 0 else np.nan

    def somatization_trychotomization(self, participant):
        val = self.get_field(participant, '4dsqstc')
        return val if val is not None and val >= 0 else np.nan

    def dichotomization_thrychotomization(self, participant):
        val = self.get_field(participant, '4dsqsdc')
        return val if val is not None and val >= 0 else np.nan