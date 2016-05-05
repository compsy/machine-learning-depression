from dataInput import SpssReader
from factories.QuestionnaireFactory import QuestionnaireFactory
from machineLearningModels import LinearRegressionModel, AsyncModelRunner, SupportVectorMachineModel, RegressionTreeModel
from models import Participant
from models.questionnaires import IDSQuestionnaire, FourDKLQuestionnaire
from outputFileCreators.SingleOutputFrameCreator import SingleOutputFrameCreator
import pickle
import numpy as np

from collections import deque


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
def create_participants(data):
    participants = {}
    for index, entry in data.iterrows():
        p = Participant.Participant(entry['pident'],
                                    entry['Sexe'],
                                    entry['Age'])
        participants[p.pident] = p

    return participants

def read_cache(cache_name):
    with open(cache_name, 'rb') as input:
        header = pickle.load(input)
        data = pickle.load(input)
        return (header, data)

def write_cache(header, data, cache_name):
  with open(cache_name, 'wb') as output:
        pickle.dump(header, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

def printHeaders(header):
    print('Available headers:')
    for col in header:
        print('\t' + col)
    print()

if __name__ == '__main__':
    with_cache = False

    spss_reader = SpssReader.SpssReader()
    N1_A100R = spss_reader.read_file("N1_A100R.sav")
    participants = create_participants(N1_A100R)
    single_output_frame_creator = SingleOutputFrameCreator()

    header, data = (None, None)
    print('Converting data to single dataframe...')
    if with_cache:
        header, data = read_cache('cache.pkl')
        printHeaders(header)
    else:
        questionnaires = QuestionnaireFactory.construct_questionnaires(spss_reader)
        data, header = (single_output_frame_creator.create_single_frame(questionnaires, participants))
        write_cache(header,data, 'cache.pkl')

    x = np.array(['ademo-gender', 'ademo-age', 'aids-somScore'])
    y = np.array(['cids-followup-somScore'])


    # Add the header to the numpy array, won't work now
    #data = map(lambda x: tuple(x), data)
    #data = np.array(deque(data), [(n, 'float64') for n in header])

    models = [
        LinearRegressionModel.LinearRegressionModel,
        SupportVectorMachineModel.SupportVectorMachineModel,
        RegressionTreeModel.RegressionTreeModel
    ]

    async_model_runner = AsyncModelRunner.AsyncModelRunner(models)
    result_queue = async_model_runner.runCalculations(data, header, x, y)

    for i in range(0, result_queue.qsize()):
        model, prediction = result_queue.get()
        model.plot(prediction)
