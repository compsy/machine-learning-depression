import os.path
import pickle

import numpy as np
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.preprocessing import normalize, scale

from data_input import spss_reader
from data_output.csv_exporter import CsvExporter
from data_output.plotters.actual_vs_prediction_plotter import ActualVsPredictionPlotter
from data_output.plotters.data_density_plotter import DataDensityPlotter
from data_output.plotters.learning_curve_plotter import LearningCurvePlotter
from data_output.plotters.roc_curve_plotter import RocCurvePlotter
from data_output.plotters.validation_curve_plotter import ValidationCurvePlotter
from data_transformers.data_preprocessor_polynomial import DataPreprocessorPolynomial
from data_transformers.output_data_cleaner import OutputDataCleaner
from data_transformers.output_data_splitter import OutputDataSplitter
from factories.questionnaire_factory import QuestionnaireFactory
from machine_learning_models import sync_model_runner
from machine_learning_models.classification.naive_bayes_model import NaiveBayesModel
from machine_learning_models.models.bagging_model import BaggingClassificationModel, BaggingModel
from machine_learning_models.models.boosting_model import BoostingClassificationModel, BoostingModel
from machine_learning_models.models.dummy_model import DummyClassifierModel, DummyRandomClassifierModel
from machine_learning_models.models.linear_regression_model import LinearRegressionModel, LogisticRegressionModel
from machine_learning_models.models.regression_tree_model import RegressionTreeModel
from machine_learning_models.models.support_vector_machine_model import SupportVectorClassificationModel
from models import participant
from output_file_creators.single_output_frame_creator import SingleOutputFrameCreator


def create_participants(data):
    participants = {}
    for index, entry in data.iterrows():
        p = participant.Participant(entry['pident'], entry['Sexe'], entry['Age'])
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


def print_header(header):
    print('\t -> Available headers:')
    for col in header:
        print('\t' + col)
    print()


def get_file_data(file_name, spss_reader, force_to_not_use_cache=False):
    header, data = (None, None)
    print('\t -> Converting data to single dataframe...')
    if not force_to_not_use_cache and os.path.isfile(file_name):
        header, data = read_cache(file_name)
        #print_header(header)
    else:
        questionnaires = QuestionnaireFactory.construct_questionnaires(spss_reader)
        data, header = (single_output_frame_creator.create_single_frame(questionnaires, participants))
        write_cache(header, data, file_name)
    return (header, data)


if __name__ == '__main__':

    # General settings
    VERBOSITY = 0

    # Should the analysis include polynomial features?
    POLYNOMIAL_FEATURES = False

    # Should we normalize?
    NORMALIZE = False
    SCALE = True

    # Classification or models?
    CLASSIFICATION = False

    FORCE_NO_CACHING = False

    spss_reader = spss_reader.SpssReader()
    single_output_frame_creator = SingleOutputFrameCreator()
    output_data_cleaner = OutputDataCleaner()
    output_data_splitter = OutputDataSplitter()
    data_preprocessor_polynomial = DataPreprocessorPolynomial()

    actual_vs_prediction_plotter = ActualVsPredictionPlotter()
    learning_curve_plotter = LearningCurvePlotter()
    validation_curve_plotter = ValidationCurvePlotter()
    roc_curve_plotter = RocCurvePlotter()
    data_density_plotter = DataDensityPlotter()

    # First read demographic data
    N1_A100R = spss_reader.read_file("N1_A100R.sav")
    participants = create_participants(N1_A100R)

    header, data = get_file_data('cache.pkl', spss_reader=spss_reader, force_to_not_use_cache=FORCE_NO_CACHING)

    print('\t -> Loaded data with %d rows and %d columns' % np.shape(data))
    # Here we select the variables to use in the prediction. The format is:
    # AB-C:
    # - A = the time of the measurement, a = intake, c = followup
    # - B = the name of the questionnaire (check QuestionnaireFactory for the correct names)
    # - C = the name of the variable. Check the name used in the <Questionnairename>questionnaire.py
    X_NAMES = np.array([  #'pident',
        # IDS - 'aids-ids09A', 'aids-ids09B', 'aids-ids09C', are NONE for nearly everyone
        'aids-somScore',
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

        # # Masq
        'amasq-positiveAffectScore',
        'amasq-negativeAffectScore',
        'amasq-somatizationScore',

        # Bai
        'abai-totalScore',
        'abai-subjectiveScaleScore',
        'abai-severityScore',
        'abai-somaticScaleScore',

        # # 4dkl
        'a4dkl-somScore',
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

        # # Cidi depression
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

        # # Cidi anxiety
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

        #'ademo-gender',
        'ademo-age'
    ])




    models = []
    if (CLASSIFICATION):
        models = [
            SupportVectorClassificationModel, LogisticRegressionModel, NaiveBayesModel, DummyClassifierModel,
            DummyRandomClassifierModel, BoostingClassificationModel, BaggingClassificationModel
        ]
        # Output columns
        Y_NAMES = np.array(['ccidi-depression-followup-majorDepressionPastSixMonths'])
    else:  # Regression
        models = [
            # KerasNnModel,
            LinearRegressionModel,
            # SupportVectorMachineModel,
            RegressionTreeModel,
            BoostingModel,
            BaggingModel
        ]
        # Output columns
        Y_NAMES = np.array(['cids-followup-somScore'])
        #Y_NAMES = np.array(['cids-followup-severity'])

    print('\t -> We will use %s as outcome.' % Y_NAMES)
    selected_header = np.append(X_NAMES, Y_NAMES)

    # Select the data we will use in the present experiment
    used_data = output_data_splitter.split(data, header, selected_header)

    # Determine which of this set are not complete
    incorrect_rows = output_data_cleaner.find_incomplete_rows(used_data, selected_header)

    print('\t -> From the loaded data %d rows are incomplete and will be removed!' % len(incorrect_rows))

    # Remove the incorrect cases
    used_data = output_data_cleaner.clean(used_data, incorrect_rows)

    # Split the dataframe into a x and y dataset.
    x_data = output_data_cleaner.clean(output_data_splitter.split(data, header, X_NAMES), incorrect_rows)

    # Logtransform the data
    # x_data[:,0] += 1
    # x_data[:,0] = np.log(x_data[:,0])


    if NORMALIZE:
        print('\t -> We are also normalizing the features')
        x_data = normalize(x_data)

    if SCALE:
        print('\t -> We are also scaling the features')
        x_data = scale(x_data)

    if POLYNOMIAL_FEATURES:
        print('\t -> We are also adding polynomial features')
        x_data = data_preprocessor_polynomial.process(x_data, X_NAMES)

    data_density_plotter.plot(x_data, X_NAMES)

    y_data = output_data_cleaner.clean(output_data_splitter.split(data, header, Y_NAMES), incorrect_rows)
    # Convert ydata 2d matrix (x * 1) to 1d array (x)
    y_data = np.ravel(y_data)

    print("\t -> The used data for the prediction has shape: %s %s" % np.shape(x_data))
    print("\t -> The values to predict have the shape: %s" % np.shape(y_data))
    # Export all used data to a CSV file

    CsvExporter.export('../exports/merged_dataframe.csv', used_data, selected_header)

    # Add the header to the numpy array, won't work now
    # data = map(lambda x: tuple(x), data)
    # data = np.array(deque(data), [(n, 'float64') for n in header])

    sync_model_runner = sync_model_runner.SyncModelRunner(models)
    fabricated_models = sync_model_runner.fabricate_models(x_data, y_data, X_NAMES, Y_NAMES, VERBOSITY)

    # Train all models, the fitted parameters will be saved inside the models
    sync_model_runner.run_calculations(fabricated_models=fabricated_models)

    # Generate learning curve plots
    #roc_curve_plotter.plot(fabricated_models)

    for model in fabricated_models:
        1
        #learning_curve_plotter.plot(model)
        # plots.append(validation_curve_plotter.plot(model))

    for model in fabricated_models:
        model.print_evaluation()
        y_train_pred = model.skmodel.predict(model.x_train)
        y_test_pred = model.skmodel.predict(model.x_test)
        actual_vs_prediction_plotter.plot_both(model, model.y_test, y_test_pred, model.y_train, y_train_pred)

