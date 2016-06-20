from models.questionnaire import Questionnaire


class CIDIAnxietyDerived(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'socialFobiaPastMonth': self.socialFobiaPastMonth,
            'socialfobiaPastSixMonths': self.socialfobiaPastSixMonths,
            'socialFobiaPastYear': self.socialFobiaPastYear,
            'socialfobiaInLifetime': self.socialfobiaInLifetime,
            'panicWithAgorafobiaPastMonth': self.panicWithAgorafobiaPastMonth,
            'panicWithAgorafobiaPastSixMonths': self.panicWithAgorafobiaPastSixMonths,
            'panicWithAgorafobiaPastYear': self.panicWithAgorafobiaPastYear,
            'panicWithAgorafobiaInLifetime': self.panicWithAgorafobiaInLifetime,
            'panicWithoutAgorafobiaPastSixMonths': self.panicWithoutAgorafobiaPastSixMonths,
            'panicWithoutAgorafobiaPastMonth': self.panicWithoutAgorafobiaPastMonth,
            'panicWithoutAgorafobiaPastYear': self.panicWithoutAgorafobiaPastYear,
            'panicWithoutAgorafobiaInLifetime': self.panicWithoutAgorafobiaInLifetime,
            'agorafobiaPastMonth': self.agorafobiaPastMonth,
            'agorafobiaPastSixMonths': self.agorafobiaPastSixMonths,
            'agorafobiaPastYear': self.agorafobiaPastYear,
            'agorafobiaInLifetime': self.agorafobiaInLifetime,
            'generalAnxietyDisorderPastMonth': self.generalAnxietyDisorderPastMonth,
            'generalAnxietyDisorderPastSixMonths': self.generalAnxietyDisorderPastSixMonths,
            'generalAnxietyDisorderPastYear': self.generalAnxietyDisorderPastYear,
            'generalAnxietyDisorderInLifetime': self.generalAnxietyDisorderInLifetime,
            'numberOfCurrentAnxietyDiagnoses': self.numberOfCurrentAnxietyDiagnoses,
            'lifetimeAnxietyDiagnosesPresent': self.lifetimeAnxietyDiagnosesPresent
        }

        super().__init__(name, filename, measurement_moment, reader, function_mapping)

    # Social fobia
    def socialFobiaPastMonth(self, participant):
        return self.getField(participant, 'anxy01')

    def socialfobiaPastSixMonths(self, participant):
        return self.getField(participant, 'anxy06')

    def socialFobiaPastYear(self, participant):
        return self.getField(participant, 'anxy11')

    def socialfobiaInLifetime(self, participant):
        return self.getField(participant, 'anxy16')

    # Panic with AgoraFobia
    def panicWithAgorafobiaPastMonth(self, participant):
        return self.getField(participant, 'anxy02')

    def panicWithAgorafobiaPastSixMonths(self, participant):
        return self.getField(participant, 'anxy07')

    def panicWithAgorafobiaPastYear(self, participant):
        return self.getField(participant, 'anxy12')

    def panicWithAgorafobiaInLifetime(self, participant):
        return self.getField(participant, 'anxy17')

    # Panic without AgoraFobia
    def panicWithoutAgorafobiaPastSixMonths(self, participant):
        return self.getField(participant, 'anxy08')

    def panicWithoutAgorafobiaPastMonth(self, participant):
        return self.getField(participant, 'anxy03')

    def panicWithoutAgorafobiaPastYear(self, participant):
        return self.getField(participant, 'anxy13')

    def panicWithoutAgorafobiaInLifetime(self, participant):
        return self.getField(participant, 'anxy18')

    # AgoraFobia
    def agorafobiaPastMonth(self, participant):
        return self.getField(participant, 'anxy04')

    def agorafobiaPastSixMonths(self, participant):
        return self.getField(participant, 'anxy09')

    def agorafobiaPastYear(self, participant):
        return self.getField(participant, 'anxy14')

    def agorafobiaInLifetime(self, participant):
        return self.getField(participant, 'anxy19')

    # Panic with General Anxiety Disorder
    def generalAnxietyDisorderPastMonth(self, participant):
        return self.getField(participant, 'anxy05')

    def generalAnxietyDisorderPastSixMonths(self, participant):
        return self.getField(participant, 'anxy10')

    def generalAnxietyDisorderPastYear(self, participant):
        return self.getField(participant, 'anxy15')

    def generalAnxietyDisorderInLifetime(self, participant):
        return self.getField(participant, 'anxy20')

    # Number of current anxiety disorders (pastSixMonths)
    def numberOfCurrentAnxietyDiagnoses(self, participant):
        return self.getField(participant, 'anxy21')

    # Lifetime Anxiety D
    def lifetimeAnxietyDiagnosesPresent(self, participant):
        return self.getField(participant, 'anxy22')
