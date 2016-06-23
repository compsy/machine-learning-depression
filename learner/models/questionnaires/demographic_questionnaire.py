from models.questionnaire import Questionnaire


class DemographicQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {'gender': self.gender, 'age': self.age}

        # educasp is not in here, as its a string
        # bcspec is not in here, as its a string
        # natspec is not in here, as its a string
        other_available_variables = [
            'frame01', 'frame02', 'area', 'educat', 'edu', 'edulvl', 'bthctry', 'natnmbr',
            'nation1', 'nation2', 'northea',
        ]

        super().__init__(name, filename, measurement_moment, reader, function_mapping, other_available_variables)

    def gender(self, participant):
        return participant.gender

    def age(self, participant):
        return participant.age

