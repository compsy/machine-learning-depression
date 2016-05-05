class Participant:
    def __init__(self, pident, sexe, age):
        self.pident = int(pident)
        self.sexe = sexe
        self.age = age
        self.questionnaires = {}

    @property
    def gender(self):
        if self.sexe == 'male':
            return 0
        return 1

    def str(self):
        return "id: %d, age: %d, sexe: %s" % (self.pident, self.age, self.sexe)

    def add_questionnaire(self, questionnaire):
        self.questionnaires[questionnaire.key] = questionnaire