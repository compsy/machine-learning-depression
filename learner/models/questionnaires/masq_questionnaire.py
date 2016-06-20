from models.questionnaire import Questionnaire
import numpy as np


class MASQQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'positiveAffectScore': self.positiveAffectScore,
            'negativeAffectScore': self.negativeAffectScore,
            'somatizationScore': self.somatizationScore
        }

        super().__init__(name, filename, measurement_moment, reader, function_mapping)

    def positiveAffectScore(self, participant):
        val = self.get_field(participant, 'masqpa')
        return val if val is not None and val >= 0 else np.nan

    def negativeAffectScore(self, participant):
        val = self.get_field(participant, 'masqna')
        return val if val is not None and val >= 0 else np.nan

    def somatizationScore(self, participant):
        val = self.get_field(participant, 'masqsa')
        return val if val is not None and val >= 0 else np.nan
