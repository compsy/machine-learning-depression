from learner.models.questionnaires import ids_questionnaire, four_dkl_distress_questionnaire, demographic_questionnaire, bai_questionnaire, \
    masq_questionnaire, cidi_depression_derived, cidi_anxiety_derived, four_dkl_physical_complaints_questionnaire, \
    neo_ffi_questionnaire, drug_usage_questionnaire, alcohol_usage_audit_questionnaire, cidi_alcohol_diagnosis, \
    mdq_questionnaire, chronic_diseaeses_questionnaire, cidi_bipolar_derived

import numpy as np

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
        questionnaires = []

        # N1_x100    DOB, age, gender, nationality and education of respondents
        questionnaires.append(demographic_questionnaire.DemographicQuestionnaire(name="demo",
                                                               filename="N1_A100R.sav",
                                                               measurement_moment='a',
                                                               reader=reader))

        # N1_x201    Drug usage
        questionnaires.append(drug_usage_questionnaire.DrugUsageQuestionnaire(name="drug-usage",
                                                                   filename='N1_A201D.SAV',
                                                                   measurement_moment='a',
                                                                   reader=reader))

        # N1_x202    Audit - Alcohol usage
        questionnaires.append(alcohol_usage_audit_questionnaire.AlcoholUsageAuditQuestionnaire(name="alcohol-usage",
                                                                              filename='N1_A202D.sav',
                                                                              measurement_moment='a',
                                                                              reader=reader))


        # N1_x203    CIDI - Alcohol diagnosis
        questionnaires.append(cidi_alcohol_diagnosis.CidiAlcoholDiagnosis(name="cidi-alcohol",
                                                                          filename='N1_A203D.sav',
                                                                          measurement_moment='a',
                                                                          reader=reader))

        # N1_x226    MASQ - Mood and Anxiety Scoring Questionnaire
        ## !! Only derived is available !!
        questionnaires.append(masq_questionnaire.MASQQuestionnaire(name="masq",
                                                                   filename='N1_A226D.SAV',
                                                                   measurement_moment='a',
                                                                   reader=reader))
        questionnaires.append(masq_questionnaire.MASQQuestionnaire(name="masq-followup",
                                                                   filename='N1_C226D.SAV',
                                                                   measurement_moment='c',
                                                                   reader=reader))


        # N1_x229    MDQ - bipolar symptoms
        questionnaires.append(mdq_questionnaire.MdqQuestionnaire(name="mdq",
                                                                 filename='N1_A229D.sav',
                                                                 measurement_moment='a',
                                                                 reader=reader))

        # N1_x232    4DKL (distress)
        questionnaires.append(four_dkl_distress_questionnaire.FourDKLQuestionnaire(name="4dkl",
                                                             filename="N1_A232R.SAV",
                                                             measurement_moment='a',
                                                             reader=reader))
        questionnaires.append(four_dkl_distress_questionnaire.FourDKLQuestionnaire(name="4dkl-followup",
                                                             filename="N1_C232R.SAV",
                                                             measurement_moment='c',
                                                             reader=reader))

        # N1_x235    IDS (Inventory Depressive Symptomatology)
        ids1 = ids_questionnaire.IDSQuestionnaire(name="ids",
                                           filename="N1_A235R.sav",
                                           measurement_moment='a',
                                           reader=reader)
        questionnaires.append(ids1)
        questionnaires.append(ids_questionnaire.IDSQuestionnaire(name="ids-followup",
                                           filename="N1_C235R.sav",
                                           measurement_moment='c',
                                           reader=reader, previous_questionnaire=ids1))


        # N1_x236    BAI (Beck Anxiety Inventory)
        ## !! Only derived is available !!
        questionnaires.append(bai_questionnaire.BAIQuestionnaire(name="bai",
                                           filename='N1_A236D.sav',
                                           measurement_moment='a',
                                           reader=reader))
        questionnaires.append(bai_questionnaire.BAIQuestionnaire(name="bai-followup",
                                           filename='N1_C236D.sav',
                                           measurement_moment='c',
                                           reader=reader))

        # N1_x240	NEO-FFI (big five personality test)
        questionnaires.append(neo_ffi_questionnaire.NeoFfiQuestionnaire(name="neoffi",
                                                                        filename='N1_A240D.sav',
                                                                        measurement_moment='a',
                                                                        reader=reader))

        # N1_x244    IDS (factor analysis subscales)

        # N1_x250    Chronic diseases/conditions
        questionnaires.append(chronic_diseaeses_questionnaire.ChronicDiseasesQuestionnaire(name="chronic-diseases",
                                                                                           filename='N1_A250D.sav',
                                                                                           measurement_moment='a',
                                                                                           reader=reader))

        # N1_x255    4DKL (physical complaints)
        questionnaires.append(
            four_dkl_physical_complaints_questionnaire.FourDKLPhysicalComplaintsQuestionnaire(name="4dkl-ph",
                                                                                              filename="N1_A255D.sav",
                                                                                              measurement_moment='a',
                                                                                              reader=reader))
        questionnaires.append(
            four_dkl_physical_complaints_questionnaire.FourDKLPhysicalComplaintsQuestionnaire(name="4dkl-ph-followup",
                                                                                              filename="N1_C255D.sav",
                                                                                              measurement_moment='c',
                                                                                              reader=reader))

        # N1_x256    CIDI - depression (raw scores + diagnoses)

        # N1_x257    CIDI - depression (derived diagnosis variables)
        ## We will be using the derived file here
        questionnaires.append(cidi_depression_derived.CIDIDepressionDerived(name="cidi-depression",
                                                      filename='N1_A257D.sav',
                                                      measurement_moment='a',
                                                      reader=reader))

        questionnaires.append(cidi_depression_derived.CIDIDepressionDerived(name="cidi-depression-followup",
                                                      filename='N1_C257D.sav',
                                                      measurement_moment='c',
                                                      reader=reader))

        # N1_x258    CIDI - anxiety (raw scores + diagnoses)
        # N1_x259    CIDI - anxiety (derived diagnoses variables)
        ## We will be using the derived file here
        questionnaires.append(cidi_anxiety_derived.CIDIAnxietyDerived(name="cidi-anxiety",
                                                filename='N1_A259D.sav',
                                                measurement_moment='a',
                                                reader=reader))

        questionnaires.append(cidi_anxiety_derived.CIDIAnxietyDerived(name="cidi-anxiety-followup",
                                                filename='N1_C259D.sav',
                                                measurement_moment='c',
                                                reader=reader))


        # N1_x261    CIDI - bipolar (raw scores + diagnoses)

        # N1_x262    CIDI- bipolar (derived diagnoses variables)
        # NOT AVAILABLE IN FIRST MEASUREMENT
        #questionnaires.append(cidi_bipolar_derived.CIDIBipolarDerived(name="cidi-bipolar",
         #                                                             filename='N1_C262D.sav',
         #                                                             measurement_moment='a',
         #                                                             reader=reader))

        # N1_x272    Course variables W1->W3
        # N1_x307    TIC-P  - Care variables
        # N1_x354    Medication use (current)
        # N1_x355    Medication use (antidepressant past 3yrs)
        # N1_x401    Blood markers
        # N1_x404    Inflammation - hsCRP/IL6
        # N1_x408    TNF-a
        # N1_x490    Saliva - measurement info
        # N1_x491    Saliva - markers (cortisol)

        return questionnaires

    @staticmethod
    def construct_x_names():
        """
          Here we select the variables to use in the prediction. The format is:
          AB-C:
          - A = the time of the measurement, a = intake, c = followup
          - B = the name of the questionnaire (check QuestionnaireFactory for the correct names)
          - C = the name of the variable. Check the name used in the <Questionnairename>questionnaire.py
        """
        return np.array([  # 'pident',
            # N1_x100    DOB, age, gender, nationality and education of respondents
            'ademo-frame01',
            'ademo-frame02',
            'ademo-area',
            'ademo-educat',
            'ademo-edu',
            'ademo-edulvl',
            'ademo-bthctry',
            'ademo-natnmbr',
            'ademo-nation1',
            'ademo-nationTwo',
            'ademo-northea',

            # N1_x201    Drug usage
            'adrug-usage-PolyDrugsUse',

            # N1_x202    Audit - Alcohol usage
            # N1_x203    CIDI - Alcohol diagnosis
            'aalcohol-usage-sumScore',
            'aalcohol-usage-medicalAdvice',

            # N1_x226    MASQ - Mood and Anxiety Scoring Questionnaire
            'amasq-positiveAffectScore',
            'amasq-negativeAffectScore',
            'amasq-somatizationScore',

            # N1_x229    MDQ - bipolar symptoms
            'amdq-MdqTotalScore',
            'amdq-MdqHirschfeldCriteriaBipolarSpectrumDisorder',
            'amdq-MdqHirschfeldCriteriaBipolarSpectrumDisorderAdaptedCriteria',

            # N1_x232    4DKL (distress)
            'a4dkl-somatizationScore',
            'a4dkl-severity',
            'a4dkl-4dkld01',
            'a4dkl-4dkld02',
            'a4dkl-4dkld03',
            'a4dkl-4dkld04',
            'a4dkl-4dkld05',
            'a4dkl-4dkld06',
            'a4dkl-4dkld07',
            'a4dkl-4dkld08',
            'a4dkl-4dkld09',
            'a4dkl-4dkld10',
            'a4dkl-4dkld11',
            'a4dkl-4dkld12',
            'a4dkl-4dkld13',
            'a4dkl-4dkld14',
            'a4dkl-4dkld15',
            'a4dkl-4dkld16',

            # N1_x235    IDS (Inventory Depressive Symptomatology)
            # IDS - 'aids-ids09A', 'aids-ids09B', 'aids-ids09C', are NONE for nearly everyone
            # 'aids-somScore',
            'aids-ids01',
            'aids-ids02',
            'aids-ids03',
            'aids-ids04',
            'aids-ids05',
            'aids-ids06',
            'aids-ids07',
            'aids-ids08',
            'aids-ids10',
            'aids-ids11',
            'aids-ids12',
            'aids-ids13',
            'aids-ids14',
            'aids-ids15',
            'aids-ids16',
            'aids-ids17',
            'aids-ids18',
            'aids-ids19',
            'aids-ids20',
            'aids-ids21',
            'aids-ids22',
            'aids-ids23',
            'aids-ids24',
            'aids-ids25',
            'aids-ids26',
            'aids-ids27',
            'aids-ids28',

            # N1_x236    BAI (Beck Anxiety Inventory)
            'abai-totalScore',
            'abai-subjectiveScaleScore',
            'abai-severityScore',
            'abai-somaticScaleScore',

            # N1_x240    NEO-FFI (big five personality test)
            'aneoffi-NeuroticismeTotalScore',
            'aneoffi-NeuroticismNegativeAffect',
            'aneoffi-NeuroticismSelfReproach',
            'aneoffi-NeuroticismAnxietyAlternative',
            'aneoffi-NeuroticismDepressionAlternative',
            'aneoffi-NeuroticismSelfreproach2Alternative',
            'aneoffi-ExtraversionTotalScore',
            'aneoffi-ExtraversionPositiveAffect',
            'aneoffi-ExtraversionSociability',
            'aneoffi-ExtraversionActivity',
            'aneoffi-OpennessTotalScore',
            'aneoffi-OpennessAestheticInterest',
            'aneoffi-OpennessIntellectualInterest',
            'aneoffi-OpennessUnconventionality',
            'aneoffi-AgreeablenessTotalScore',
            'aneoffi-AgreeablenessNonantagonasticOrientation',
            'aneoffi-AgreeablenessProsocialOrientation',
            'aneoffi-ConscientiousnessTotalScore',
            'aneoffi-ConscientiousnessOrderliness',
            'aneoffi-ConscientiousnessGoalStriving',
            'aneoffi-ConscientiousnessDependability',

            # N1_x244    IDS (factor analysis subscales)

            # N1_x250    Chronic diseases/conditions
            'achronic-diseases-numberOfChronicDiseases',
            'achronic-diseases-numberOfChronicDiseasesUnderTreatment',

            # N1_x255    4DKL (physical complaints)
            'a4dkl-ph-somatizationSumScore',
            'a4dkl-ph-somatizationTrychotomization',
            'a4dkl-ph-dichotomizationThrychotomization',

            # N1_x256    CIDI - depression (raw scores + diagnoses)

            # N1_x257    CIDI - depression (derived diagnosis variables)
            'acidi-depression-minorDepressionPastMonth',
            'acidi-depression-majorDepressionPastMonth',
            'acidi-depression-majorDepressionPastSixMonths',
            'acidi-depression-majorDepressionPastYear',
            'acidi-depression-majorDepressionLifetime',
            'acidi-depression-dysthymiaPastmonth',
            'acidi-depression-dysthymiaPastSixMonths',
            'acidi-depression-dysthymiaPastYear',
            'acidi-depression-dysthymiaLifetime',
            'acidi-depression-numberOfCurrentDepressionDiagnoses',
            'acidi-depression-hasLifetimeDepressionDiagnoses',
            'acidi-depression-categoriesForLifetimeDepressionDiagnoses',
            # 'acidi-depression-numberOfMajorDepressionEpisodes',
            # 'acidi-depression-majorDepressionType',

            # N1_x258    CIDI - anxiety (raw scores + diagnoses)

            # N1_x259    CIDI - anxiety (derived diagnoses variables)
            'acidi-anxiety-socialFobiaPastMonth',
            'acidi-anxiety-socialfobiaPastSixMonths',
            'acidi-anxiety-socialFobiaPastYear',
            'acidi-anxiety-socialfobiaInLifetime',
            'acidi-anxiety-panicWithAgorafobiaPastMonth',
            'acidi-anxiety-panicWithAgorafobiaPastSixMonths',
            'acidi-anxiety-panicWithAgorafobiaPastYear',
            'acidi-anxiety-panicWithAgorafobiaInLifetime',
            'acidi-anxiety-panicWithoutAgorafobiaPastSixMonths',
            'acidi-anxiety-panicWithoutAgorafobiaPastMonth',
            'acidi-anxiety-panicWithoutAgorafobiaPastYear',
            'acidi-anxiety-panicWithoutAgorafobiaInLifetime',
            'acidi-anxiety-agorafobiaPastMonth',
            'acidi-anxiety-agorafobiaPastSixMonths',
            'acidi-anxiety-agorafobiaPastYear',
            'acidi-anxiety-agorafobiaInLifetime',
            'acidi-anxiety-generalAnxietyDisorderPastMonth',
            'acidi-anxiety-generalAnxietyDisorderPastSixMonths',
            'acidi-anxiety-generalAnxietyDisorderPastYear',
            'acidi-anxiety-generalAnxietyDisorderInLifetime',
            'acidi-anxiety-numberOfCurrentAnxietyDiagnoses',
            'acidi-anxiety-lifetimeAnxietyDiagnosesPresent',

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

            'ademo-gender',
            'ademo-age'
        ])