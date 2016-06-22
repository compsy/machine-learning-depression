from ..questionnaire import Questionnaire
import numpy as np


class MASQQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'positiveAffectScore': self.positive_affect_score,
            'negativeAffectScore': self.negative_affect_score,
            'somatizationScore': self.somatization_score
        }

        super().__init__(name, filename, measurement_moment, reader, function_mapping)

    def positive_affect_score(self, participant):
        val = self.get_field(participant, 'masqpa')
        return val if val is not None and val >= 0 else np.nan

    def negative_affect_score(self, participant):
        val = self.get_field(participant, 'masqna')
        return val if val is not None and val >= 0 else np.nan

    def somatization_score(self, participant):
        val = self.get_field(participant, 'masqsa')
        return val if val is not None and val >= 0 else np.nan
