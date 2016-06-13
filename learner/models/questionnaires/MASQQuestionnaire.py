from models.Questionnaire import Questionnaire


class MASQQuestionnaire(Questionnaire):
    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'positiveAffectScore': self.positiveAffectScore,
            'negativeAffectScore': self.negativeAffectScore,
            'somatizationScore':   self.somatizationScore
        }

        super().__init__(name,filename,measurement_moment,reader, function_mapping)

    def positiveAffectScore(self, participant):
        return self.getField(participant, 'masqpa')

    def negativeAffectScore(self, participant):
        return self.getField(participant, 'masqna')

    def somatizationScore(self, participant):
        return self.getField(participant, 'masqsa')
