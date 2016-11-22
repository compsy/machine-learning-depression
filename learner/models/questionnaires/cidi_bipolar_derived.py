from ..questionnaire import Questionnaire
import numpy as np


class CIDIBipolarDerived(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        # All variables are dichotomous
        function_mapping = {
            'bipolarDisorderIPastmonth': self.bipolar_disorder_i_pastmonth,
            'bipolarDisorderIIPastmonth': self.bipolar_disorder_i_i_pastmonth,
            'bipolarDisorderIPast6months': self.bipolar_disorder_i_past6months,
            'bipolarDisorderIIPast6months': self.bipolar_disorder_i_i_past6months,
            'bipolarDisorderIPastyear': self.bipolar_disorder_i_pastyear,
            'bipolarDisorderIIPastyear': self.bipolar_disorder_i_i_pastyear,
            'bipolarDisorderIInlifetime': self.bipolar_disorder_i_inlifetime,
            'bipolarDisorderIIInlifetime': self.bipolar_disorder_i_i_inlifetime,
            'currentBipolardiagnosespresent': self.current_bipolardiagnosespresent,
            'lifetimeBipolardiagnosespresent': self.lifetime_bipolardiagnosespresent
        }

        other_available_variables = []

        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

    # Social fobia

    def bipolar_disorder_i_pastmonth(self, participant):
        val = self.get_field(participant, 'bip01', force_lower_case= True)
        return val if val is not None and val >= 0 else np.nan

    def bipolar_disorder_i_i_pastmonth(self, participant):
        val = self.get_field(participant, 'bip02', force_lower_case= True)
        return val if val is not None and val >= 0 else np.nan

    def bipolar_disorder_i_past6months(self, participant):
        val = self.get_field(participant, 'bip03', force_lower_case= True)
        return val if val is not None and val >= 0 else np.nan

    def bipolar_disorder_i_i_past6months(self, participant):
        val = self.get_field(participant, 'bip04', force_lower_case= True)
        return val if val is not None and val >= 0 else np.nan

    def bipolar_disorder_i_pastyear(self, participant):
        val = self.get_field(participant, 'bip05', force_lower_case= True)
        return val if val is not None and val >= 0 else np.nan

    def bipolar_disorder_i_i_pastyear(self, participant):
        val = self.get_field(participant, 'bip06', force_lower_case= True)
        return val if val is not None and val >= 0 else np.nan

    def bipolar_disorder_i_inlifetime(self, participant):
        val = self.get_field(participant, 'bip07', force_lower_case= True)
        return val if val is not None and val >= 0 else np.nan

    def bipolar_disorder_i_i_inlifetime(self, participant):
        val = self.get_field(participant, 'bip08', force_lower_case= True)
        return val if val is not None and val >= 0 else np.nan

    def current_bipolardiagnosespresent(self, participant):
        val = self.get_field(participant, 'bip09', force_lower_case= True)
        return val if val is not None and val >= 0 else np.nan

    def lifetime_bipolardiagnosespresent(self, participant):
        val = self.get_field(participant, 'bip10', force_lower_case=False)
        return val if val is not None and val >= 0 else np.nan
