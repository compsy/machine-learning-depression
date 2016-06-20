from models.questionnaire import Questionnaire
import numpy as np

class IDSQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {'somScore': self.somScore, 'severity': self.severity}

        super().__init__(name, filename, measurement_moment, reader, function_mapping)
        self.variables_for_som_score = [
            'ids01', 'ids02', 'ids03', 'ids04', 'ids05', 'ids06', 'ids07', 'ids08', 'ids09A', 'ids09B', 'ids09C',
            'ids10', 'ids11', 'ids12', 'ids13', 'ids14', 'ids15', 'ids16', 'ids17', 'ids18', 'ids19', 'ids20', 'ids21',
            'ids22', 'ids23', 'ids24', 'ids25', 'ids26', 'ids27'
        ]

    def somScore(self, participant):
        dat = self.getRow(participant)
        tot = 0
        for name in self.variables_for_som_score:
            q_name = self.variableName(name)
            if q_name in dat and dat[q_name] >= 0:
                tot += dat[q_name]
        return tot if tot > 0 else np.nan

    def severity(self, participant):
        score = self.somScore(participant)
        if np.isnan(score):
            return np.nan
        elif score<= 13:
            return 0
        elif score <= 25:
            return 1
        elif score <= 38:
            return 2
        elif score <= 48:
            return 3
        elif score <= 84:
            return 4

        return None
