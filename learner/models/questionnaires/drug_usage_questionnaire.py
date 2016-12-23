from ..questionnaire import Questionnaire
import numpy as np


class DrugUsageQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        # -1 = "Q1 not reurned"
        # 0 = a valid score. (0 = "0 kinds of drugs")

        function_mapping = {'PolyDrugsUse': self.poly_drug_use}

        other_available_variables = []

        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

    # Social fobia

    def poly_drug_use(self, participant):
        val = self.get_field(participant, 'polyd', force_lower_case=True)
        return val if val is not None and val >= 0 else np.nan
