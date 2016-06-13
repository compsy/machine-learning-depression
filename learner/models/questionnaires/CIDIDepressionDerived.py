from models.Questionnaire import Questionnaire


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

        super().__init__(name,filename,measurement_moment,reader, function_mapping)


    # Depression
    def minorDepressionPastMonth(self, participant):
        return self.getField(self, participant, 'acidep01')

    def majorDepressionPastMonth(self, participant):
        return self.getField(self, participant, 'acidep03')

    def majorDepressionPastSixMonths(self, participant):
        return self.getField(self, participant, 'acidep05')

    def majorDepressionPastYear(self, participant):
        return self.getField(self, participant, 'acidep07')

    def majorDepressionLifetime(self, participant):
        return self.getField(self, participant, 'acidep09')


    # Dysthymia
    def dysthymiaPastmonth(self, participant):
        return self.getField(self, participant, 'acidep02')

    def dysthymiaPastSixMonths(self, participant):
        return self.getField(self, participant, 'acidep04')

    def dysthymiaPastYear(self, participant):
        return self.getField(self, participant, 'acidep06')

    def dysthymiaLifetime(self, participant):
        return self.getField(self, participant, 'acidep08')


    # number of current depression diagnoses (past 6 months)
    def numberOfCurrentDepressionDiagnoses(self, participant):
        return self.getField(self, participant, 'acidep10')

    def hasLifetimeDepressionDiagnoses(self, participant):
        return self.getField(self, participant, 'acidep11')

    def categoriesForLifetimeDepressionDiagnoses(self, participant):
        return self.getField(self, participant, 'acidep12')


    # of MDD episodes
    def numberOfMajorDepressionEpisodes(self, participant):
        return self.getField(self, participant, 'acidep13')

    def majorDepressionType(self, participant):
        return self.getField(self, participant, 'acidep14')

