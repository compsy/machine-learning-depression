from models.questionnaires import ids_questionnaire, four_dkl_questionnaire, demographic_questionnaire, bai_questionnaire, \
    masq_questionnaire, cidi_depression_derived, cidi_anxiety_derived

# Dataset    Description
# N1_x100    DOB, age, gender, nationality and education of respondents
# N1_x201    Drug usage
# N1_x202    Audit - Alcohol usage
# N1_x203    CIDI - Alcohol diagnosis
# N1_x226    MASQ - Mood and Anxiety Scoring Questionnaire
# N1_x229    MDQ - bipolar symptoms
# N1_x232    4DKL (distress)
# N1_x235    IDS (Inventory Depressive Symptomatology)
# N1_x236    BAI (Beck Anxiety Inventory)
# N1_x240    NEO-FFI (big five personality test)
# N1_x244    IDS (factor analysis subscales)
# N1_x250    Chronic diseases/conditions
# N1_x255    4DKL (physical complaints)
# N1_x256    CIDI - depression (raw scores + diagnoses)
# N1_x257    CIDI - depression (derived diagnosis variables)
# N1_x258    CIDI - anxiety (raw scores + diagnoses)
# N1_x259    CIDI - anxiety (derived diagnoses variables)
# N1_x261    CIDI - bipolar (raw scores + diagnoses)
# N1_x262    CIDI- bipolar (derived diagnoses variables)
# N1_x272    Course variables W1->W3
# N1_x307    TIC-P  - Care variables
# N1_x354    Medication use (current)
# N1_x355    Medication use (antidepressant past 3yrs)
# N1_x401    Blood markers
# N1_x404    Inflammation - hsCRP/IL6
# N1_x408    TNF-a
# N1_x490    Saliva - measurement info
# N1_x491    Saliva - markers (cortisol)

# A = first wave
# C = second wave
# R = Raw
# D = derived


class QuestionnaireFactory:

    @staticmethod
    def construct_questionnaires(reader):
        questionnaires = [
            demographic_questionnaire.DemographicQuestionnaire(name="demo",
                                                               filename="N1_A100R.sav",
                                                               measurement_moment='a',
                                                               reader=reader),

            # N1_x235    IDS (Inventory Depressive Symptomatology)
            ids_questionnaire.IDSQuestionnaire(name="ids",
                                               filename="N1_A235R.sav",
                                               measurement_moment='a',
                                               reader=reader),
            ids_questionnaire.IDSQuestionnaire(name="ids-followup",
                                               filename="N1_C235R.sav",
                                               measurement_moment='c',
                                               reader=reader),

            # N1_x255    4DKL (physical complaints)
            four_dkl_questionnaire.FourDKLQuestionnaire(name="4dkl",
                                                        filename="N1_A232R.sav",
                                                        measurement_moment='a',
                                                        reader=reader),
            four_dkl_questionnaire.FourDKLQuestionnaire(name="4dkl-followup",
                                                        filename="N1_C232R.sav",
                                                        measurement_moment='c',
                                                        reader=reader),

            # N1_x236    BAI (Beck Anxiety Inventory)
            ## !! Only derived is available !!
            bai_questionnaire.BAIQuestionnaire(name="bai",
                                               filename='N1_A236D.sav',
                                               measurement_moment='a',
                                               reader=reader),
            bai_questionnaire.BAIQuestionnaire(name="bai-followup",
                                               filename='N1_C236D.sav',
                                               measurement_moment='c',
                                               reader=reader),

            # N1_x226    MASQ - Mood and Anxiety Scoring Questionnaire
            ## !! Only derived is available !!
            masq_questionnaire.MASQQuestionnaire(name="masq",
                                                 filename='N1_A226D.sav',
                                                 measurement_moment='a',
                                                 reader=reader),
            masq_questionnaire.MASQQuestionnaire(name="masq-followup",
                                                 filename='N1_C226D.sav',
                                                 measurement_moment='c',
                                                 reader=reader),

            # N1_x257    CIDI - depression (derived diagnosis variables)
            ## We will be using the derived file here
            cidi_depression_derived.CIDIDepressionDerived(name="cidi-depression",
                                                          filename='N1_A257D.sav',
                                                          measurement_moment='a',
                                                          reader=reader),
            cidi_depression_derived.CIDIDepressionDerived(name="cidi-depression-followup",
                                                          filename='N1_C257D.sav',
                                                          measurement_moment='c',
                                                          reader=reader),

            # N1_x259    CIDI - anxiety (derived diagnoses variables)
            ## We will be using the derived file here
            cidi_anxiety_derived.CIDIAnxietyDerived(name="cidi-anxiety",
                                                    filename='N1_A259D.sav',
                                                    measurement_moment='a',
                                                    reader=reader),
            cidi_anxiety_derived.CIDIAnxietyDerived(name="cidi-anxiety-followup",
                                                    filename='N1_C259D.sav',
                                                    measurement_moment='c',
                                                    reader=reader)
        ]
        return questionnaires