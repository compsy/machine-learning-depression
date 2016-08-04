from ..questionnaire import Questionnaire
import numpy as np


class AuditQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):

        #-3 = "Q1 not reurned"
        # --1 = "Too many missings"
        # 0 = a valid score. (0 = "R Does not drink")
        function_mapping = {
            'sumScore': self.sum_score,
            'medicalAdvice': self.medical_advice
        }
        other_available_variables = []
        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

    def sum_score(self, participant):
        val = self.get_field(participant, 'auditsc')
        return val if val is not None and val >= 0 else np.nan

    def medical_advice(self, participant):
        val = self.get_field(participant, 'auditma')
        return val if val is not None and val >= 0 else np.nan
