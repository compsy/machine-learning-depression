from ..questionnaire import Questionnaire
import numpy as np


class MdqQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):

        #-3 = "MDQ not assessed"
        # -1 = "Too many missings"
        # 0 = a valid score.
        function_mapping = {
            'MdqTotalScore':
            self.mdq_total_score,
            'MdqHirschfeldCriteriaBipolarSpectrumDisorder':
            self.mdq_hirschfeld_criteria_bipolar_spectrum_disorder,
            'MdqHirschfeldCriteriaBipolarSpectrumDisorderAdaptedCriteria':
            self.mdq_hirschfeld_criteria_bipolar_spectrum_disorder_adapted_criteria
        }
        other_available_variables = []
        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

    def mdq_total_score(self, participant):
        val = self.get_field(participant, 'mdqtotal')
        return val if val is not None and val >= 0 else np.nan

    def mdq_hirschfeld_criteria_bipolar_spectrum_disorder(self, participant):
        val = self.get_field(participant, 'mdqposa')
        return val if val is not None and val >= 0 else np.nan

    def mdq_hirschfeld_criteria_bipolar_spectrum_disorder_adapted_criteria(self, participant):
        val = self.get_field(participant, 'mdqposa2')
        return val if val is not None and val >= 0 else np.nan
