from models.questionnaires import IDSQuestionnaire, FourDKLQuestionnaire, DemographicQuestionnaire


class QuestionnaireFactory:
    @staticmethod
    def construct_questionnaires(reader):
        questionnaires = [
            DemographicQuestionnaire.DemographicQuestionnaire(name="demo", filename="N1_A100R.sav",
                                                      measurement_moment='a', reader=reader),

            IDSQuestionnaire.IDSQuestionnaire(name="ids", filename="N1_A235R.sav", measurement_moment='a',
                                              reader=reader),
            IDSQuestionnaire.IDSQuestionnaire(name="ids-followup", filename="N1_C235R.sav", measurement_moment='c',
                                              reader=reader),
            FourDKLQuestionnaire.FourDKLQuestionnaire(name="4dkl", filename="N1_A232R.sav", measurement_moment='a',
                                                      reader=reader),
            FourDKLQuestionnaire.FourDKLQuestionnaire(name="4dkl-followup", filename="N1_C232R.sav",
                                                      measurement_moment='c', reader=reader),
        ]
        return questionnaires
