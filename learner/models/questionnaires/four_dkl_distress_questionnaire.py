from ..questionnaire import Questionnaire
import numpy as np
from data_output.std_logger import L


class FourDKLQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {'somScore': self.som_score}

        other_available_variables = [
            '4dkld01', '4dkld02', '4dkld03', '4dkld04', '4dkld05', '4dkld06', '4dkld07', '4dkld08', '4dkld09',
            '4dkld10', '4dkld11', '4dkld12', '4dkld13', '4dkld14', '4dkld15', '4dkld16'
        ]

        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

        # TODO: check what the correct variables are for calculating the sum score
        self.variables_for_som_score = [
            '4dkld01', '4dkld02', '4dkld03', '4dkld04', '4dkld05', '4dkld06', '4dkld07', '4dkld08', '4dkld09',
            '4dkld10', '4dkld11', '4dkld12', '4dkld13', '4dkld14', '4dkld15', '4dkld16'
        ]

    def som_score(self, participant):
        # L.warn('Before using this function, test whether the correct variables are used for calculating this score!')
        dat = self.get_row(participant)

        # If there are no values > 0, we return nan
        tot = np.nan

        for name in self.variables_for_som_score:
            q_name = self.variable_name(name)
            # We can check here for values > 0 since the NESDA dataset uses values from 1 - 5?
            if q_name in dat and dat[q_name] > 0:
                if (np.isnan(tot)): tot = 0
                tot += dat[q_name] - 1

        return tot

    def severity(self, participant):
        score = self.som_score(participant)
        if np.isnan(score):
            return np.nan
        elif score <= 10:
            return 0
        elif score <= 16:
            return 1
        elif score <= 32:
            return 2

        return 2
