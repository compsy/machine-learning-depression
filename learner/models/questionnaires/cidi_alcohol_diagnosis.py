from ..questionnaire import Questionnaire
import numpy as np


class CidiAlcoholDiagnosis(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'DsmivDiagnosisAlcoholD': self.dsmiv_diagnosis_alcohol,
            'AgeOnsetAlcoholD': self.age_onset_alcohol,
            'RecencyAlcoholD': self.recency_alcohol,
            'DsmivDiagnosisAlcoholAbuse': self.dsmiv_diagnosis_alcohol_abuse,
            'AgeOnsetAlcoholAbuse': self.age_onset_alcohol_abuse,
            'RecencyAlcoholAbuse': self.recency_alcohol_abuse,
            'AlcoholDiagnoseStatus': self.alcohol_diagnose_status
        }
        other_available_variables = []
        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)


    def dsmiv_diagnosis_alcohol(self, participant):
        # 0 = "Diag Indeterminate", is this considered a missing?
        val = self.get_field(participant,'d30390')
        return val if val is not None and val >= 1 else np.nan

    def age_onset_alcohol(self, participant):
        # 998 = "Refusal"
        # 999 = "Don't Know"
        val = self.get_field(participant,'d30390ao')
        return val if val is not None and val >= 0 and val < 998 else np.nan

    def recency_alcohol(self, participant):
        val = self.get_field(participant,'d30390re')
        return val if val is not None and val >= 0 else np.nan

    def dsmiv_diagnosis_alcohol_abuse(self, participant):
        val = self.get_field(participant,'d30500')
        return val if val is not None and val >= 0 else np.nan

    def age_onset_alcohol_abuse(self, participant):
        # 998 = "Refusal"
        # 999 = "Don't Know"
        val = self.get_field(participant,'d30500ao')
        return val if val is not None and val >= 0 and val < 998 else np.nan

    def recency_alcohol_abuse(self, participant):
        val = self.get_field(participant,'d30500re')
        return val if val is not None and val >= 0 else np.nan

    def alcohol_diagnose_status(self, participant):
        # -1 = "Diagnose indeterminate". I think this is considered a missing
        val = self.get_field(participant,'lcversl')
        return val if val is not None and val >= 0 else np.nan
