class Participant:

    def __init__(self, pident, sexe, age):
        self.pident = int(pident)
        self.sexe = sexe
        self.age = age
        self.questionnaires = {}

    @property
    def gender(self):
        if self.sexe == 'male' or self.sexe == 1:
            return 0
        if self.sexe == 'female' or self.sexe == 2:
            return 1
        return None

    def str(self):
        return "id: %d, age: %d, sexe: %s" % (self.pident, self.age, self.sexe)

    def add_questionnaire(self, questionnaire):
        self.questionnaires[questionnaire.key] = questionnaire
