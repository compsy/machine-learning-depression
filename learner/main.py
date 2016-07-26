import os.path
import pickle
import numpy as np
import random
from data_output.std_logger import L
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.preprocessing import normalize, scale
from mpi4py import MPI

from data_input import spss_reader
from data_output.csv_exporter import CsvExporter
from data_output.plotters.actual_vs_prediction_plotter import ActualVsPredictionPlotter
from data_output.plotters.data_density_plotter import DataDensityPlotter
from data_output.plotters.learning_curve_plotter import LearningCurvePlotter
from data_output.plotters.roc_curve_plotter import RocCurvePlotter
from data_output.plotters.validation_curve_plotter import ValidationCurvePlotter
from data_output.plotters.confusion_matrix_plotter import ConfusionMatrixPlotter

from data_transformers.data_preprocessor_polynomial import DataPreprocessorPolynomial
from data_transformers.output_data_cleaner import OutputDataCleaner
from data_transformers.output_data_splitter import OutputDataSplitter
from data_transformers.variable_transformer import VariableTransformer
from factories.questionnaire_factory import QuestionnaireFactory
from machine_learning_models.sync_model_runner import SyncModelRunner
from machine_learning_models.distributed_model_runner import DistributedModelRunner

from machine_learning_models.classification.naive_bayes_model import NaiveBayesModel
from machine_learning_models.models.bagging_model import BaggingClassificationModel, BaggingModel
from machine_learning_models.models.boosting_model import BoostingClassificationModel, BoostingModel
from machine_learning_models.models.dummy_model import DummyClassifierModel, DummyRandomClassifierModel
from machine_learning_models.models.regression_model import LinearRegressionModel, LogisticRegressionModel
from machine_learning_models.models.tree_model import RegressionTreeModel, ClassificationTreeModel
from machine_learning_models.models.support_vector_machine_model import SupportVectorRegressionModel, SupportVectorClassificationModel
# from machine_learning_models.models.keras_nn_model import KerasNnModel, KerasNnClassificationModel

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
    L.info('Available headers:')
    for col in header:
        L.info('\t' + col)
    L.br()


def get_file_data(file_name, spss_reader, force_to_not_use_cache=False):
    header, data = (None, None)
    L.info('Converting data to single dataframe...')
    if not force_to_not_use_cache and os.path.isfile(file_name):
        header, data = read_cache(file_name)
        #print_header(header)
    else:
        questionnaires = QuestionnaireFactory.construct_questionnaires(spss_reader)
        data, header = (single_output_frame_creator.create_single_frame(questionnaires, participants))
        write_cache(header, data, file_name)
    return (header, data)

def calculate_true_false_ratio(y_data):
    trues = 0
    falses = 0
    for i in y_data:
        if i == 0:
            falses += 1
        if i == 1:
            trues += 1

    return (trues / (trues + falses)) * 100


if __name__ == '__main__':
    L.setup()
    random.seed(42)

    # General settings
    VERBOSITY = 0
    HPC = True
    # Should the analysis include polynomial features?
    POLYNOMIAL_FEATURES = False

    # Should we normalize?
    NORMALIZE = False
    SCALE = True

    # Classification or models?
    CLASSIFICATION = True

    FORCE_NO_CACHING = True

    # Here we select the variables to use in the prediction. The format is:
    # AB-C:
    # - A = the time of the measurement, a = intake, c = followup
    # - B = the name of the questionnaire (check QuestionnaireFactory for the correct names)
    # - C = the name of the variable. Check the name used in the <Questionnairename>questionnaire.py
    X_NAMES = np.array([  # 'pident',
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
        #'a4dkl-somScore',
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

        'a4dkl-ph-somatizationSumScore',
        'a4dkl-ph-somatizationTrychotomization',
        'a4dkl-ph-dichotomizationThrychotomization',

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

        # 'ademo-gender',
        'ademo-age'
    ])

    ##### Define the models we should run
    models = []
    if (CLASSIFICATION):
        models = [
            SupportVectorClassificationModel,
            LogisticRegressionModel,
            # BaggingClassificationModel,
            BoostingClassificationModel,
            NaiveBayesModel,
            DummyClassifierModel,
            # DummyRandomClassifierModel,
            ClassificationTreeModel
            #KerasNnClassificationModel
        ]
        # Output columns
        Y_NAMES = np.array(['ccidi-depression-followup-majorDepressionPastSixMonths'])
    else:  # Regression
        models = [
            # KerasNnModel,
            LinearRegressionModel, SupportVectorRegressionModel, RegressionTreeModel, BoostingModel,
            BaggingModel
        ]
        # Output columns
        Y_NAMES = np.array(['cids-followup-somScore'])
        #Y_NAMES = np.array(['cids-followup-severity'])

    spss_reader = spss_reader.SpssReader()
    single_output_frame_creator = SingleOutputFrameCreator()
    output_data_cleaner = OutputDataCleaner()
    output_data_splitter = OutputDataSplitter()
    data_preprocessor_polynomial = DataPreprocessorPolynomial()

    actual_vs_prediction_plotter = ActualVsPredictionPlotter()
    learning_curve_plotter = LearningCurvePlotter()
    validation_curve_plotter = ValidationCurvePlotter()
    roc_curve_plotter = RocCurvePlotter()
    confusion_matrix_plotter = ConfusionMatrixPlotter()
    data_density_plotter = DataDensityPlotter()
    variable_transformer = VariableTransformer(X_NAMES)

    # First read demographic data
    N1_A100R = spss_reader.read_file("N1_A100R.sav")
    participants = create_participants(N1_A100R)
    header, data = get_file_data('cache.pkl', spss_reader=spss_reader, force_to_not_use_cache=FORCE_NO_CACHING)

    L.info('We have %d participants in the inital dataset' % len(participants.keys()))
    L.info('Loaded data with %d rows and %d columns' % np.shape(data))
    L.info('We will use %s as outcome.' % Y_NAMES)

    selected_header = np.append(X_NAMES, Y_NAMES)

    # Select the data we will use in the present experiment (used_data = both x and y)
    used_data = output_data_splitter.split(data, header, selected_header)

    # Determine which of this set are not complete
    incorrect_rows = output_data_cleaner.find_incomplete_rows(used_data, selected_header)

    L.info('From the loaded data %d rows are incomplete and will be removed!' % len(incorrect_rows))

    # Remove the incorrect cases
    used_data = output_data_cleaner.clean(used_data, incorrect_rows)

    # Split the dataframe into a x and y dataset.
    x_data = output_data_splitter.split(used_data, selected_header, X_NAMES)

    # Logtransform the data
    # variable_transformer.log_transform(x_data, 'aids-somScore')

    if NORMALIZE:
        L.info('We are also normalizing the features')
        x_data = normalize(x_data)

    if SCALE:
        L.info('We are also scaling the features')
        x_data = scale(x_data)

    if POLYNOMIAL_FEATURES:
        L.info('We are also adding polynomial features')
        x_data = data_preprocessor_polynomial.process(x_data, X_NAMES)


    y_data = output_data_cleaner.clean(output_data_splitter.split(data, header, Y_NAMES), incorrect_rows)

    if CLASSIFICATION:
        L.info('In the output set, %0.2f percent is true' % calculate_true_false_ratio(y_data))

    L.info("The used data for the prediction has shape: %s %s" % np.shape(x_data))
    L.info("The values to predict have the shape: %s %s" % np.shape(y_data))

    # Convert ydata 2d matrix (x * 1) to 1d array (x). Needed for the classifcation things
    y_data = np.ravel(y_data)


    if HPC:
        model_runner = DistributedModelRunner(models)
    else:
        model_runner = SyncModelRunner(models)

    fabricated_models = model_runner.fabricate_models(x_data, y_data, X_NAMES, Y_NAMES, VERBOSITY)

    # Train all models, the fitted parameters will be saved inside the models
    is_root, fabricated_models = model_runner.run_calculations(fabricated_models=fabricated_models)

    # Kill all worker nodes
    if not is_root:
        exit(0)

    # Plot an overview of the density estimations of the variables used in the actual model calculation.
    data_density_plotter.plot(x_data, X_NAMES)

    # Export all used data to a CSV file
    CsvExporter.export('../exports/merged_dataframe.csv', used_data, selected_header)

    # Generate learning curve plots
    if CLASSIFICATION:
        roc_curve_plotter.plot(fabricated_models)

    for model in fabricated_models:
        1
        #learning_curve_plotter.plot(model)
        #validation_curve_plotter.plot(model, variable_to_validate='n_estimators')

    for model in fabricated_models:
        model.print_evaluation()
        y_train_pred = model.skmodel.predict(model.x_train)
        y_test_pred = model.skmodel.predict(model.x_test)

        if CLASSIFICATION:
            confusion_matrix_plotter.plot(model, model.y_test, y_test_pred)
        else:
            actual_vs_prediction_plotter.plot_both(model, model.y_test, y_test_pred, model.y_train, y_train_pred)
