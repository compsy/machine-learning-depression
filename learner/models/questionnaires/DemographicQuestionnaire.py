from models.Questionnaire import Questionnaire


class DemographicQuestionnaire(Questionnaire):

    def __init__(self, name, filename, measurement_moment, reader):
        function_mapping = {'gender': self.gender, 'age': self.age}

        super().__init__(name, filename, measurement_moment, reader,
                         function_mapping)

    def gender(self, participant):
        return participant.gender

    def age(self, participant):
        return participant.age

    def levelOfEducation(self, participant):
        return self.getField(participant, 'dulvl')
