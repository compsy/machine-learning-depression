from ..questionnaire import Questionnaire
import numpy as np


class NeoFfiQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        #-2 = "Q1 not returned"
        #-1 = "Too many missings"

        function_mapping = {
            'NeuroticismeTotalScore': self.neuroticisme_total_score,
            'NeuroticismNegativeAffect': self.neuroticism_negative_affect,
            'NeuroticismSelfReproach': self.neuroticism_self_reproach,
            'NeuroticismAnxietyAlternative': self.neuroticism_anxiety_alternative,
            'NeuroticismDepressionAlternative': self.neuroticism_depression_alternative,
            'NeuroticismSelfreproach2Alternative': self.neuroticism_selfreproach2_alternative,
            'ExtraversionTotalScore': self.extraversion_total_score,
            'ExtraversionPositiveAffect': self.extraversion_positive_affect,
            'ExtraversionSociability': self.extraversion_sociability,
            'ExtraversionActivity': self.extraversion_activity,
            'OpennessTotalScore': self.openness_total_score,
            'OpennessAestheticInterest': self.openness_aesthetic_interest,
            'OpennessIntellectualInterest': self.openness_intellectual_interest,
            'OpennessUnconventionality': self.openness_unconventionality,
            'AgreeablenessTotalScore': self.agreeableness_total_score,
            'AgreeablenessNonantagonasticOrientation': self.agreeableness_nonantagonastic_orientation,
            'AgreeablenessProsocialOrientation': self.agreeableness_prosocial_orientation,
            'ConscientiousnessTotalScore': self.conscientiousness_total_score,
            'ConscientiousnessOrderliness': self.conscientiousness_orderliness,
            'ConscientiousnessGoalStriving': self.conscientiousness_goal_striving,
            'ConscientiousnessDependability': self.conscientiousness_dependability
        }
        other_available_variables = []
        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

    def neuroticisme_total_score(self, participant):
        val = self.get_field(participant, 'neurot')
        return val if val is not None and val >= 0 else np.nan

    def neuroticism_negative_affect(self, participant):
        val = self.get_field(participant, 'neuro2')
        return val if val is not None and val >= 0 else np.nan

    def neuroticism_self_reproach(self, participant):
        val = self.get_field(participant, 'neuro3')
        return val if val is not None and val >= 0 else np.nan

    def neuroticism_anxiety_alternative(self, participant):
        val = self.get_field(participant, 'neuro4')
        return val if val is not None and val >= 0 else np.nan

    def neuroticism_depression_alternative(self, participant):
        val = self.get_field(participant, 'neuro5')
        return val if val is not None and val >= 0 else np.nan

    def neuroticism_selfreproach2_alternative(self, participant):
        val = self.get_field(participant, 'neuro6')
        return val if val is not None and val >= 0 else np.nan

    def extraversion_total_score(self, participant):
        val = self.get_field(participant, 'extrave')
        return val if val is not None and val >= 0 else np.nan

    def extraversion_positive_affect(self, participant):
        val = self.get_field(participant, 'extrav2')
        return val if val is not None and val >= 0 else np.nan

    def extraversion_sociability(self, participant):
        val = self.get_field(participant, 'extrav3')
        return val if val is not None and val >= 0 else np.nan

    def extraversion_activity(self, participant):
        val = self.get_field(participant, 'extrav4')
        return val if val is not None and val >= 0 else np.nan

    def openness_total_score(self, participant):
        val = self.get_field(participant, 'openes')
        return val if val is not None and val >= 0 else np.nan

    def openness_aesthetic_interest(self, participant):
        val = self.get_field(participant, 'opene2')
        return val if val is not None and val >= 0 else np.nan

    def openness_intellectual_interest(self, participant):
        val = self.get_field(participant, 'opene3')
        return val if val is not None and val >= 0 else np.nan

    def openness_unconventionality(self, participant):
        val = self.get_field(participant, 'opene4')
        return val if val is not None and val >= 0 else np.nan

    def agreeableness_total_score(self, participant):
        val = self.get_field(participant, 'agreeab')
        return val if val is not None and val >= 0 else np.nan

    def agreeableness_nonantagonastic_orientation(self, participant):
        val = self.get_field(participant, 'agreea2')
        return val if val is not None and val >= 0 else np.nan

    def agreeableness_prosocial_orientation(self, participant):
        val = self.get_field(participant, 'agreea3')
        return val if val is not None and val >= 0 else np.nan

    def conscientiousness_total_score(self, participant):
        val = self.get_field(participant, 'conscie')
        return val if val is not None and val >= 0 else np.nan

    def conscientiousness_orderliness(self, participant):
        val = self.get_field(participant, 'consci2')
        return val if val is not None and val >= 0 else np.nan

    def conscientiousness_goal_striving(self, participant):
        val = self.get_field(participant, 'consci3')
        return val if val is not None and val >= 0 else np.nan

    def conscientiousness_dependability(self, participant):
        val = self.get_field(participant, 'consci4')
        return val if val is not None and val >= 0 else np.nan
