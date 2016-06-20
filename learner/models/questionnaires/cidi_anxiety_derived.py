from models.questionnaire import Questionnaire
import numpy as np

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
        val = self.get_field(participant, 'anxy01')
        return val if val is not None and val >= 0 else np.nan

    def socialfobiaPastSixMonths(self, participant):
        val = self.get_field(participant, 'anxy06')
        return val if val is not None and val >= 0 else np.nan

    def socialFobiaPastYear(self, participant):
        val = self.get_field(participant, 'anxy11')
        return val if val is not None and val >= 0 else np.nan

    def socialfobiaInLifetime(self, participant):
        val = self.get_field(participant, 'anxy16')
        return val if val is not None and val >= 0 else np.nan

    # Panic with AgoraFobia
    def panicWithAgorafobiaPastMonth(self, participant):
        val = self.get_field(participant, 'anxy02')
        return val if val is not None and val >= 0 else np.nan

    def panicWithAgorafobiaPastSixMonths(self, participant):
        val = self.get_field(participant, 'anxy07')
        return val if val is not None and val >= 0 else np.nan

    def panicWithAgorafobiaPastYear(self, participant):
        val = self.get_field(participant, 'anxy12')
        return val if val is not None and val >= 0 else np.nan

    def panicWithAgorafobiaInLifetime(self, participant):
        val = self.get_field(participant, 'anxy17')
        return val if val is not None and val >= 0 else np.nan

    # Panic without AgoraFobia
    def panicWithoutAgorafobiaPastSixMonths(self, participant):
        val = self.get_field(participant, 'anxy08')
        return val if val is not None and val >= 0 else np.nan

    def panicWithoutAgorafobiaPastMonth(self, participant):
        val = self.get_field(participant, 'anxy03')
        return val if val is not None and val >= 0 else np.nan

    def panicWithoutAgorafobiaPastYear(self, participant):
        val = self.get_field(participant, 'anxy13')
        return val if val is not None and val >= 0 else np.nan

    def panicWithoutAgorafobiaInLifetime(self, participant):
        val = self.get_field(participant, 'anxy18')
        return val if val is not None and val >= 0 else np.nan

    # AgoraFobia
    def agorafobiaPastMonth(self, participant):
        val = self.get_field(participant, 'anxy04')
        return val if val is not None and val >= 0 else np.nan

    def agorafobiaPastSixMonths(self, participant):
        val = self.get_field(participant, 'anxy09')
        return val if val is not None and val >= 0 else np.nan

    def agorafobiaPastYear(self, participant):
        val = self.get_field(participant, 'anxy14')
        return val if val is not None and val >= 0 else np.nan

    def agorafobiaInLifetime(self, participant):
        val = self.get_field(participant, 'anxy19')
        return val if val is not None and val >= 0 else np.nan

    # Panic with General Anxiety Disorder
    def generalAnxietyDisorderPastMonth(self, participant):
        val = self.get_field(participant, 'anxy05')
        return val if val is not None and val >= 0 else np.nan

    def generalAnxietyDisorderPastSixMonths(self, participant):
        val = self.get_field(participant, 'anxy10')
        return val if val is not None and val >= 0 else np.nan

    def generalAnxietyDisorderPastYear(self, participant):
        val = self.get_field(participant, 'anxy15')
        return val if val is not None and val >= 0 else np.nan

    def generalAnxietyDisorderInLifetime(self, participant):
        val = self.get_field(participant, 'anxy20')
        return val if val is not None and val >= 0 else np.nan

    # Number of current anxiety disorders (pastSixMonths)
    def numberOfCurrentAnxietyDiagnoses(self, participant):
        val = self.get_field(participant, 'anxy21')
        return val if val is not None and val >= 0 else np.nan

    # Lifetime Anxiety D
    def lifetimeAnxietyDiagnosesPresent(self, participant):
        val = self.get_field(participant, 'anxy22')
        return val if val is not None and val >= 0 else np.nan
