from models.questionnaire import Questionnaire
import numpy as np


class CIDIDepressionDerived(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'minorDepressionPastMonth': self.minorDepressionPastMonth,
            'majorDepressionPastMonth': self.majorDepressionPastMonth,
            'majorDepressionPastSixMonths': self.majorDepressionPastSixMonths,
            'majorDepressionPastYear': self.majorDepressionPastYear,
            'majorDepressionLifetime': self.majorDepressionLifetime,
            'dysthymiaPastmonth': self.dysthymiaPastmonth,
            'dysthymiaPastSixMonths': self.dysthymiaPastSixMonths,
            'dysthymiaPastYear': self.dysthymiaPastYear,
            'dysthymiaLifetime': self.dysthymiaLifetime,
            'numberOfCurrentDepressionDiagnoses': self.numberOfCurrentDepressionDiagnoses,
            'hasLifetimeDepressionDiagnoses': self.hasLifetimeDepressionDiagnoses,
            'categoriesForLifetimeDepressionDiagnoses': self.categoriesForLifetimeDepressionDiagnoses,
            'numberOfMajorDepressionEpisodes': self.numberOfMajorDepressionEpisodes,
            'majorDepressionType': self.majorDepressionType
        }

        super().__init__(name, filename, measurement_moment, reader, function_mapping)

    # Depression
    def minorDepressionPastMonth(self, participant):
        val = self.getField(participant, 'cidep01')
        return val if val is not None and val >= 0 else np.nan

    def majorDepressionPastMonth(self, participant):
        val = self.getField(participant, 'cidep03')
        return val if val is not None and val >= 0 else np.nan

    def majorDepressionPastSixMonths(self, participant):
        val = self.getField(participant, 'cidep05')
        return val if val is not None and val >= 0 else np.nan

    def majorDepressionPastYear(self, participant):
        val = self.getField(participant, 'cidep07')
        return val if val is not None and val >= 0 else np.nan

    def majorDepressionLifetime(self, participant):
        val = self.getField(participant, 'cidep09')
        return val if val is not None and val >= 0 else np.nan

    # Dysthymia
    def dysthymiaPastmonth(self, participant):
        val = self.getField(participant, 'cidep02')
        return val if val is not None and val >= 0 else np.nan

    def dysthymiaPastSixMonths(self, participant):
        val = self.getField(participant, 'cidep04')
        return val if val is not None and val >= 0 else np.nan

    def dysthymiaPastYear(self, participant):
        val = self.getField(participant, 'cidep06')
        return val if val is not None and val >= 0 else np.nan

    def dysthymiaLifetime(self, participant):
        val = self.getField(participant, 'cidep08')
        return val if val is not None and val >= 0 else np.nan

    # number of current depression diagnoses (past 6 months)
    def numberOfCurrentDepressionDiagnoses(self, participant):
        val = self.getField(participant, 'cidep10')
        return val if val is not None and val >= 0 else np.nan

    def hasLifetimeDepressionDiagnoses(self, participant):
        val = self.getField(participant, 'cidep11')
        return val if val is not None and val >= 0 else np.nan

    def categoriesForLifetimeDepressionDiagnoses(self, participant):
        val = self.getField(participant, 'cidep12')
        return val if val is not None and val >= 0 else np.nan

    # of MDD episodes
    def numberOfMajorDepressionEpisodes(self, participant):
        val = self.getField(participant, 'cidep13')
        return val if val is not None and val >= 0 else np.nan

    def majorDepressionType(self, participant):
        val = self.getField(participant, 'cidep14')
        return val if val is not None and val >= 0 else np.nan
