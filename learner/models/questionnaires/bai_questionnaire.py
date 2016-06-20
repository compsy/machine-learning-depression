from models.questionnaire import Questionnaire
import numpy as np


class BAIQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'totalScore': self.totalScore,
            'subjectiveScaleScore': self.subjectiveScaleScore,
            'severityScore': self.severityScore,
            'somaticScaleScore': self.somaticScaleScore
        }

        super().__init__(name, filename, measurement_moment, reader, function_mapping)

    def totalScore(self, participant):
        val = self.get_field(participant, 'baiscal')
        return val if val is not None and val >= 0 else np.nan

    def subjectiveScaleScore(self, participant):
        val = self.get_field(participant, 'baisub')
        return val if val is not None and val >= 0 else np.nan

    def severityScore(self, participant):
        val = self.get_field(participant, 'baisev')
        return val if val is not None and val >= 0 else np.nan

    def somaticScaleScore(self, participant):
        val = self.get_field(participant, 'baisom')
        return val if val is not None and val >= 0 else np.nan
