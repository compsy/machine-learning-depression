from ..questionnaire import Questionnaire
import numpy as np


class CIDIAnxietyDerived(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'socialFobiaPastMonth': self.social_fobia_past_month,
            'socialfobiaPastSixMonths': self.social_fobia_past_six_months,
            'socialFobiaPastYear': self.social_fobia_past_year,
            'socialfobiaInLifetime': self.social_fobia_in_lifetime,
            'panicWithAgorafobiaPastMonth': self.panic_with_agora_fobia_past_month,
            'panicWithAgorafobiaPastSixMonths': self.panic_with_agora_fobia_past_six_months,
            'panicWithAgorafobiaPastYear': self.panic_with_agora_fobia_past_year,
            'panicWithAgorafobiaInLifetime': self.panic_with_agora_fobia_in_lifetime,
            'panicWithoutAgorafobiaPastSixMonths': self.panic_without_agora_fobia_past_six_months,
            'panicWithoutAgorafobiaPastMonth': self.panic_without_agora_fobia_past_month,
            'panicWithoutAgorafobiaPastYear': self.panic_without_agora_fobia_past_year,
            'panicWithoutAgorafobiaInLifetime': self.panic_without_agora_fobia_in_lifetime,
            'agorafobiaPastMonth': self.agora_fobia_past_month,
            'agorafobiaPastSixMonths': self.agora_fobia_past_six_months,
            'agorafobiaPastYear': self.agora_fobia_past_year,
            'agorafobiaInLifetime': self.agora_fobia_in_lifetime,
            'generalAnxietyDisorderPastMonth': self.general_anxiety_disorder_past_month,
            'generalAnxietyDisorderPastSixMonths': self.general_anxiety_disorder_past_six_months,
            'generalAnxietyDisorderPastYear': self.general_anxiety_disorder_past_year,
            'generalAnxietyDisorderInLifetime': self.general_anxiety_disorder_in_lifetime,
            'numberOfCurrentAnxietyDiagnoses': self.number_of_current_anxiety_diagnoses,
            'lifetimeAnxietyDiagnosesPresent': self.lifetime_anxiety_diagnoses_present
        }

        super().__init__(name, filename, measurement_moment, reader, function_mapping)

    # Social fobia
    def social_fobia_past_month(self, participant):
        val = self.get_field(participant, 'anxy01')
        return val if val is not None and val >= 0 else np.nan

    def social_fobia_past_six_months(self, participant):
        val = self.get_field(participant, 'anxy06')
        return val if val is not None and val >= 0 else np.nan

    def social_fobia_past_year(self, participant):
        val = self.get_field(participant, 'anxy11')
        return val if val is not None and val >= 0 else np.nan

    def social_fobia_in_lifetime(self, participant):
        val = self.get_field(participant, 'anxy16')
        return val if val is not None and val >= 0 else np.nan

    # Panic with AgoraFobia
    def panic_with_agora_fobia_past_month(self, participant):
        val = self.get_field(participant, 'anxy02')
        return val if val is not None and val >= 0 else np.nan

    def panic_with_agora_fobia_past_six_months(self, participant):
        val = self.get_field(participant, 'anxy07')
        return val if val is not None and val >= 0 else np.nan

    def panic_with_agora_fobia_past_year(self, participant):
        val = self.get_field(participant, 'anxy12')
        return val if val is not None and val >= 0 else np.nan

    def panic_with_agora_fobia_in_lifetime(self, participant):
        val = self.get_field(participant, 'anxy17')
        return val if val is not None and val >= 0 else np.nan

    # Panic without AgoraFobia
    def panic_without_agora_fobia_past_six_months(self, participant):
        val = self.get_field(participant, 'anxy08')
        return val if val is not None and val >= 0 else np.nan

    def panic_without_agora_fobia_past_month(self, participant):
        val = self.get_field(participant, 'anxy03')
        return val if val is not None and val >= 0 else np.nan

    def panic_without_agora_fobia_past_year(self, participant):
        val = self.get_field(participant, 'anxy13')
        return val if val is not None and val >= 0 else np.nan

    def panic_without_agora_fobia_in_lifetime(self, participant):
        val = self.get_field(participant, 'anxy18')
        return val if val is not None and val >= 0 else np.nan

    # AgoraFobia
    def agora_fobia_past_month(self, participant):
        val = self.get_field(participant, 'anxy04')
        return val if val is not None and val >= 0 else np.nan

    def agora_fobia_past_six_months(self, participant):
        val = self.get_field(participant, 'anxy09')
        return val if val is not None and val >= 0 else np.nan

    def agora_fobia_past_year(self, participant):
        val = self.get_field(participant, 'anxy14')
        return val if val is not None and val >= 0 else np.nan

    def agora_fobia_in_lifetime(self, participant):
        val = self.get_field(participant, 'anxy19')
        return val if val is not None and val >= 0 else np.nan

    # Panic with General Anxiety Disorder
    def general_anxiety_disorder_past_month(self, participant):
        val = self.get_field(participant, 'anxy05')
        return val if val is not None and val >= 0 else np.nan

    def general_anxiety_disorder_past_six_months(self, participant):
        val = self.get_field(participant, 'anxy10')
        return val if val is not None and val >= 0 else np.nan

    def general_anxiety_disorder_past_year(self, participant):
        val = self.get_field(participant, 'anxy15')
        return val if val is not None and val >= 0 else np.nan

    def general_anxiety_disorder_in_lifetime(self, participant):
        val = self.get_field(participant, 'anxy20')
        return val if val is not None and val >= 0 else np.nan

    # Number of current anxiety disorders (pastSixMonths)
    def number_of_current_anxiety_diagnoses(self, participant):
        val = self.get_field(participant, 'anxy21')
        return val if val is not None and val >= 0 else np.nan

    # Lifetime Anxiety D
    def lifetime_anxiety_diagnoses_present(self, participant):
        val = self.get_field(participant, 'anxy22')
        return val if val is not None and val >= 0 else np.nan
