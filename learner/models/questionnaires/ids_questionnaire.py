from ..questionnaire import Questionnaire
import numpy as np
from learner.data_output.std_logger import L


class IDSQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader, previous_questionnaire=None):
        self.previous_questionnaire = previous_questionnaire
        function_mapping = {
            'somScore': self.som_score,
            'severity': self.severity,
            'twice_depression': self.twice_depression
        }

        other_available_variables = [
            'ids01', 'ids02', 'ids03', 'ids04', 'ids05', 'ids06', 'ids07', 'ids08', 'ids09a', 'ids09b', 'ids09c',
            'ids10', 'ids11', 'ids12', 'ids13', 'ids14', 'ids15', 'ids16', 'ids17', 'ids18', 'ids19', 'ids20', 'ids21',
            'ids22', 'ids23', 'ids24', 'ids25', 'ids26', 'ids27', 'ids28'
        ]

        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

        # http://www.ids-qids.org/index2.html#SCORING
        self.variables_for_som_score = [
            'ids01', 'ids02', 'ids03', 'ids04', 'ids05', 'ids06', 'ids07', 'ids08', 'ids09a', 'ids09b', 'ids09c',
            'ids10', 'ids11', 'ids12', 'ids13', 'ids14', 'ids15', 'ids16', 'ids17', 'ids18', 'ids19', 'ids20', 'ids21',
            'ids22', 'ids23', 'ids24', 'ids25', 'ids26', 'ids27', 'ids28'
        ]

    def twice_depression(self, participant):
        result = np.NaN
        # Threhold of 2 is used to comply with VanBorkulo2015
        threshold = 2
        if self.previous_questionnaire is not None:
            if self.severity(participant) > threshold and self.previous_questionnaire.severity(participant) > threshold:
                result = 1
            else:
                result = 0
        return result

    def som_score(self, participant):
        # L.warn('Check if the sumscore calculation is correct this way')
        dat = self.get_row(participant)

        # If there are no values > 0, we return nan
        tot = np.nan

        for name in self.variables_for_som_score:
            q_name = self.variable_name(name, force_lower_case=False)
            # We can check here for values > 0 since the NESDA dataset uses values from 1 - 5?
            if q_name in dat and dat[q_name] > 0:
                if (np.isnan(tot)): tot = 0
                tot += dat[q_name] - 1

        return tot

    def severity(self, participant):
        score = self.som_score(participant)
        if np.isnan(score):
            return np.nan
        elif score <= 13:
            return 0
        elif score <= 25:
            return 1
        elif score <= 38:
            return 2
        elif score <= 48:
            return 3
        elif score <= 84:
            return 4

        return 4
