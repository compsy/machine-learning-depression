from dataCleaners.OutputDataFrameCleaner import OutputDataFrameCleaner
from dataInput import SpssReader
from dataOutput.CsvExporter import CsvExporter
from factories.QuestionnaireFactory import QuestionnaireFactory
from machineLearningModels import LinearRegressionModel, AsyncModelRunner, SupportVectorMachineModel, RegressionTreeModel
from models import Participant
from models.questionnaires import IDSQuestionnaire, FourDKLQuestionnaire
from outputFileCreators.SingleOutputFrameCreator import SingleOutputFrameCreator
import pickle
import numpy as np

from collections import deque


def create_participants(data):
    participants = {}
    for index, entry in data.iterrows():
        p = Participant.Participant(entry['pident'], entry['Sexe'],
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

    # First read demographic data
    N1_A100R = spss_reader.read_file("N1_A100R.sav")
    participants = create_participants(N1_A100R)

    single_output_frame_creator = SingleOutputFrameCreator()
    outputDataFrameCleaner = OutputDataFrameCleaner()

    header, data = (None, None)
    print('Converting data to single dataframe...')
    if with_cache:
        header, data = read_cache('cache.pkl')
        printHeaders(header)
    else:
        questionnaires = QuestionnaireFactory.construct_questionnaires(
            spss_reader)
        data, header = (single_output_frame_creator.create_single_frame(
            questionnaires, participants))
        write_cache(header, data, 'cache.pkl')

    # Here we select the variables to use in the prediction. The format is:
    # AB-C:
    # - A = the time of the measurement, a = intake, c = followup
    # - B = the name of the questionnaire (check QuestionnaireFactory for the correct names)
    # - C = the name of the variable. Check the name used in the <Questionnairename>Questionnaire.py
    X = np.array(['pident', 'ademo-gender', 'ademo-age', 'aids-somScore',
                  'amasq-positiveAffectScore', 'amasq-negativeAffectScore',
                  'amasq-somatizationScore', 'abai-totalScore',
                  'abai-subjectiveScaleScore', 'abai-severityScore',
                  'abai-somaticScaleScore', 'a4dkl-somScore', 'a4dkl-severity',
                  'acidi-depression-majorDepressionLifetime',
                  'acidi-depression-dysthymiaLifetime',
                  'acidi-anxiety-socialfobiaInLifetime',
                  'acidi-anxiety-panicWithAgorafobiaInLifetime',
                  'acidi-anxiety-panicWithoutAgorafobiaInLifetime'])

    Y = np.array(['cids-followup-somScore'])

    selected_header = np.append(X, Y)

    used_data = outputDataFrameCleaner.clean(data, selected_header, header)
    CsvExporter.export('../exports/merged_dataframe.csv', used_data,
                       selected_header)

    # Add the header to the numpy array, won't work now
    #data = map(lambda x: tuple(x), data)
    #data = np.array(deque(data), [(n, 'float64') for n in header])

    models = [
        LinearRegressionModel.LinearRegressionModel,
        SupportVectorMachineModel.SupportVectorMachineModel,
        RegressionTreeModel.RegressionTreeModel
    ]

    async_model_runner = AsyncModelRunner.AsyncModelRunner(models, workers=8)
    result_queue = async_model_runner.runCalculations(used_data,
                                                      selected_header, X, Y)

    for i in range(0, result_queue.qsize()):
        model, prediction = result_queue.get()
        #model.plot(prediction)
