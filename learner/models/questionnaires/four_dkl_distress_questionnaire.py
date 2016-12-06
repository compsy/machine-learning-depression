from ..questionnaire import Questionnaire
import numpy as np
from data_output.std_logger import L


class FourDKLQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {'somatizationScore': self.somatization_score}

        other_available_variables = [
            '4dkld01', '4dkld02', '4dkld03', '4dkld04', '4dkld05', '4dkld06', '4dkld07', '4dkld08', '4dkld09',
            '4dkld10', '4dkld11', '4dkld12', '4dkld13', '4dkld14', '4dkld15', '4dkld16'
        ]

        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

        self.variables_for_somatization_score = [
            '4dkld01', '4dkld02', '4dkld03', '4dkld04', '4dkld05', '4dkld06', '4dkld07', '4dkld08', '4dkld09',
            '4dkld10', '4dkld11', '4dkld12', '4dkld13', '4dkld14', '4dkld15', '4dkld16'
        ]

    def somatization_score(self, participant):
        # http://webapps.hag-intranet.nl/rooster/bijlagen/Wetenschap%204DKL.pdf
        # https://www.telepsy.nl/sites/default/files/Vierdimensionale%20Klachtenlijst.pdf

        dat = self.get_row(participant)

        # If there are no values > 0, we return nan
        tot = np.nan

        for name in self.variables_for_somatization_score:
            q_name = self.variable_name(name, force_lower_case=False)
            # We can check here for values > 0 since the NESDA dataset uses values from 1 - 5?
            if q_name in dat and dat[q_name] > 0:
                if (np.isnan(tot)): tot = 0
                if dat[q_name] == 1:
                    # Nee
                    tot += 0
                elif dat[q_name] == 2:
                    # Soms
                    tot += 1
                elif dat[q_name] >= 3:
                    # Regelmatig, vaak of voordurend
                    tot += 2

        return tot

    def severity(self, participant):
        # http://webapps.hag-intranet.nl/rooster/bijlagen/Wetenschap%204DKL.pdf
        # https://www.telepsy.nl/sites/default/files/Vierdimensionale%20Klachtenlijst.pdf

        score = self.somatization_score(participant)
        if np.isnan(score):
            return np.nan
        elif score <= 10:
            return 0
        elif score <= 20:
            # Matig verhoogd
            return 1
        elif score <= 32:
            # Sterk verhoogd
            return 2

        return 2
