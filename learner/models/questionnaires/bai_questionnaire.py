from ..questionnaire import Questionnaire
import numpy as np


class BAIQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'totalScore': self.total_score,
            'subjectiveScaleScore': self.subjective_scale_score,
            'severityScore': self.severity_score,
            'somaticScaleScore': self.somatic_scale_score
        }

        super().__init__(name, filename, measurement_moment, reader, function_mapping)

    def total_score(self, participant):
        val = self.get_field(participant, 'baiscal')
        return val if val is not None and val >= 0 else np.nan

    def subjective_scale_score(self, participant):
        val = self.get_field(participant, 'baisub')
        return val if val is not None and val >= 0 else np.nan

    def severity_score(self, participant):
        val = self.get_field(participant, 'baisev')
        return val if val is not None and val >= 0 else np.nan

    def somatic_scale_score(self, participant):
        val = self.get_field(participant, 'baisom')
        return val if val is not None and val >= 0 else np.nan
