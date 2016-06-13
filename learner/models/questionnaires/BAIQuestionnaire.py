from models.Questionnaire import Questionnaire


class BAIQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'totalScore': self.totalScore,
            'subjectiveScaleScore': self.subjectiveScaleScore,
            'severityScore': self.severityScore,
            'somaticScaleScore': self.somaticScaleScore
        }

        super().__init__(name, filename, measurement_moment, reader,
                         function_mapping)

    def totalScore(self, participant):
        return self.getField(participant, 'baiscal')

    def subjectiveScaleScore(self, participant):
        return self.getField(participant, 'baisub')

    def severityScore(self, participant):
        return self.getField(participant, 'baisev')

    def somaticScaleScore(self, participant):
        return self.getField(participant, 'baisom')
