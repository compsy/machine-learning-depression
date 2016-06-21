from ..questionnaire import Questionnaire
import numpy as np


class CIDIDepressionDerived(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {
            'minorDepressionPastMonth': self.minor_depression_past_month,
            'majorDepressionPastMonth': self.major_depression_past_month,
            'majorDepressionPastSixMonths': self.major_depression_past_six_months,
            'majorDepressionPastYear': self.major_depression_past_year,
            'majorDepressionLifetime': self.major_depression_lifetime,
            'dysthymiaPastmonth': self.dysthymia_past_month,
            'dysthymiaPastSixMonths': self.dysthymia_past_six_months,
            'dysthymiaPastYear': self.dysthymia_past_year,
            'dysthymiaLifetime': self.dysthymia_lifetime,
            'numberOfCurrentDepressionDiagnoses': self.number_of_current_depression_diagnoses,
            'hasLifetimeDepressionDiagnoses': self.has_lifetime_depression_diagnoses,
            'categoriesForLifetimeDepressionDiagnoses': self.categories_for_lifetime_depression_diagnoses,
            'numberOfMajorDepressionEpisodes': self.number_of_major_depression_episodes,
            'majorDepressionType': self.major_depression_type
        }

        super().__init__(name, filename, measurement_moment, reader, function_mapping)

    # Depression
    def minor_depression_past_month(self, participant):
        val = self.get_field(participant, 'cidep01')
        return val if val is not None and val >= 0 else np.nan

    def major_depression_past_month(self, participant):
        val = self.get_field(participant, 'cidep03')
        return val if val is not None and val >= 0 else np.nan

    def major_depression_past_six_months(self, participant):
        val = self.get_field(participant, 'cidep05')
        return val if val is not None and val >= 0 else np.nan

    def major_depression_past_year(self, participant):
        val = self.get_field(participant, 'cidep07')
        return val if val is not None and val >= 0 else np.nan

    def major_depression_lifetime(self, participant):
        val = self.get_field(participant, 'cidep09')
        return val if val is not None and val >= 0 else np.nan

    # Dysthymia
    def dysthymia_past_month(self, participant):
        val = self.get_field(participant, 'cidep02')
        return val if val is not None and val >= 0 else np.nan

    def dysthymia_past_six_months(self, participant):
        val = self.get_field(participant, 'cidep04')
        return val if val is not None and val >= 0 else np.nan

    def dysthymia_past_year(self, participant):
        val = self.get_field(participant, 'cidep06')
        return val if val is not None and val >= 0 else np.nan

    def dysthymia_lifetime(self, participant):
        val = self.get_field(participant, 'cidep08')
        return val if val is not None and val >= 0 else np.nan

    # number of current depression diagnoses (past 6 months)
    def number_of_current_depression_diagnoses(self, participant):
        val = self.get_field(participant, 'cidep10')
        return val if val is not None and val >= 0 else np.nan

    def has_lifetime_depression_diagnoses(self, participant):
        val = self.get_field(participant, 'cidep11')
        return val if val is not None and val >= 0 else np.nan

    def categories_for_lifetime_depression_diagnoses(self, participant):
        val = self.get_field(participant, 'cidep12')
        return val if val is not None and val >= 0 else np.nan

    # of MDD episodes
    def number_of_major_depression_episodes(self, participant):
        val = self.get_field(participant, 'cidep13')
        return val if val is not None and val >= 0 else np.nan

    def major_depression_type(self, participant):
        val = self.get_field(participant, 'cidep14')
        return val if val is not None and val >= 0 else np.nan
