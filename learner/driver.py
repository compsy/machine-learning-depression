import os.path
import pickle
import numpy as np
import random
from mpi4py import MPI
from data_output.std_logger import L
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.preprocessing import normalize, scale

from data_input.spss_reader import SpssReader
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

from machine_learning_models.models.naive_bayes_model import NaiveBayesModel
from machine_learning_models.models.bagging_model import BaggingClassificationModel, BaggingModel
from machine_learning_models.models.boosting_model import BoostingClassificationModel, BoostingModel
from machine_learning_models.models.dummy_model import DummyClassifierModel, DummyRandomClassifierModel
from machine_learning_models.models.regression_model import LinearRegressionModel, LogisticRegressionModel
from machine_learning_models.models.tree_model import RegressionTreeModel, ClassificationTreeModel
from machine_learning_models.models.support_vector_machine_model import SupportVectorRegressionModel, \
    SupportVectorClassificationModel
# from machine_learning_models.models.keras_nn_model import KerasNnModel, KerasNnClassificationModel

from models import participant
from output_file_creators.single_output_frame_creator import SingleOutputFrameCreator


class Driver:
    """ The main class that runs the application
    Parameters
    ----------
    verbosity : int, default=0
        A number representing the verbosity of the model calculation.
        1 is on, 0 is off.
    hpc : boolean, default=False
        Should the application run each of the models on a separate node?
        This should not be combined with distributed gridsearch.
    hpc_log : boolean, default=True
        Consider the logging is done on an HPC? If so, the logging of the
        slaves is turned off
    polynomial_features : boolean, default=False
        should the data also include polynomial features?
    normalize : boolean, default=False
        should the input be normalized?
    scale : boolean, default=True
        should the data be scaled (mean and SD)
    classification : boolean, default=True
        True = we are performing classification, False = regression
    forc_no_caching : boolean, default=False
        By default we cache the loaded data, should we force busting this cache?
    """

    def __init__(self,
                 verbosity=0,
                 hpc=False,
                 hpc_log=True,
                 polynomial_features=False,
                 normalize=False,
                 scale=True,
                 classification=True,
                 force_no_caching=False):
        if(hpc): print('Hello from node %d' % MPI.COMM_WORLD.Get_rank())
        random.seed(42)

        L.setup(hpc_log)

        self.VERBOSITY = verbosity
        self.HPC = hpc
        self.POLYNOMIAL_FEATURES = polynomial_features
        self.NORMALIZE = normalize
        self.SCALE = scale
        self.CLASSIFICATION = classification
        self.FORCE_NO_CACHING = force_no_caching

        x_names = self.construct_x_names()

        self.actual_vs_prediction_plotter = ActualVsPredictionPlotter()
        self.learning_curve_plotter = LearningCurvePlotter()
        self.validation_curve_plotter = ValidationCurvePlotter()
        self.roc_curve_plotter = RocCurvePlotter()
        self.confusion_matrix_plotter = ConfusionMatrixPlotter()
        self.data_density_plotter = DataDensityPlotter()

        self.spss_reader = SpssReader()
        self.single_output_frame_creator = SingleOutputFrameCreator()
        self.output_data_cleaner = OutputDataCleaner()
        self.output_data_splitter = OutputDataSplitter()
        self.data_preprocessor_polynomial = DataPreprocessorPolynomial()
        self.variable_transformer = VariableTransformer(x_names)

        ##### Define the models we should run
        classification_models = []
        classification_models.append(ClassificationTreeModel)
        classification_models.append(SupportVectorClassificationModel)
        classification_models.append(BoostingClassificationModel)
        classification_models.append(LogisticRegressionModel)
        classification_models.append(NaiveBayesModel)
        classification_models.append(DummyClassifierModel)
        classification_models.append(DummyRandomClassifierModel)
        # classification_models.append(BaggingClassificationModel)
        # classification_models.append(KerasNnClassificationModel)

        regression_models = []
        # regressionmodels.append(KerasNnModel)
        regression_models.append(LinearRegressionModel)
        regression_models.append(SupportVectorRegressionModel)
        regression_models.append(RegressionTreeModel)
        regression_models.append(BoostingModel)
        # regression_models.append(BaggingModel)

        # Output columns
        classification_y_names = np.array(['ccidi-depression-followup-majorDepressionPastSixMonths'])
        classification_y_names = np.array(['cids-followup-twice_depression'])
        regression_y_names = np.array(['cids-followup-somScore'])

        participants = self.create_participants()
        header, data = self.get_file_data('cache.pkl',
                                          participants=participants,
                                          force_to_not_use_cache=self.FORCE_NO_CACHING)

        # L.info('We have %d participants in the inital dataset' % len(participants.keys()))
        x_data, classification_y_data, used_data, selected_header = self.get_usable_data(data, header, x_names,
                                                                                         classification_y_names)


        is_root, classification_fabricated_models = self.calculate(classification_models, x_data, classification_y_data,
                                                                   x_names,
                                                                   classification_y_names)

        x_data, regression_y_data, used_data, selected_header = self.get_usable_data(data, header, x_names,
                                                                                     regression_y_names)
        is_root, regression_fabricated_models = self.calculate(regression_models, x_data, regression_y_data, x_names,
                                                               regression_y_names)

        # Kill all worker nodes
        if hpc and MPI.COMM_WORLD.Get_rank() > 0:
            L.info('Byebye from node %d' % MPI.COMM_WORLD.Get_rank(), force=True)
            exit(0)

        # Plot an overview of the density estimations of the variables used in the actual model calculation.
        self.create_descriptives(participants, x_data, x_names)
        self.create_output(classification_fabricated_models, classification_y_data, used_data, selected_header,
                           model_type='classification')
        self.create_output(regression_fabricated_models, regression_y_data, used_data, selected_header,
                           model_type='regression')

    def create_descriptives(self, participants, x_data, x_names):
        ages = []
        genders = []
        for participant in participants:
            genders.append(participant.gender)
            ages.append(participant.age)
        
        gender_output = np.bincount(genders)
        if len(gender_output is not 2):
            L.warn('There are more than 2 types of people in the DB')
            L.warn(genders)

        gender_output = (gender_output[0], gender_output[1])
        age_output = (len(participants), np.average(ages), np.median(ages), np.std(ages))

        L.info('The participants (%d) have an average age of %0.2f, median %0.2f, sd %0.2f' % ages_output)
        L.info('The participants are %0.f percent male (%0.2f percent female)' % gender_output)
        self.data_density_plotter.plot(x_data, x_names)


    def get_usable_data(self, data, header, x_names, y_names):
        L.info('Loaded data with %d rows and %d columns' % np.shape(data))
        L.info('We will use %s as outcome.' % y_names)


        selected_header = np.append(x_names, y_names)

        # Select the data we will use in the present experiment (used_data = both x and y)
        used_data = self.output_data_splitter.split(data, header, selected_header)
        # Determine which of this set are not complete
        incorrect_rows = self.output_data_cleaner.find_incomplete_rows(used_data, selected_header)

        L.info('From the loaded data %d rows are incomplete and will be removed!' % len(incorrect_rows))

        # Remove the incorrect cases
        used_data = self.output_data_cleaner.clean(used_data, incorrect_rows)

        # Split the dataframe into a x and y dataset.
        x_data = self.output_data_splitter.split(used_data, selected_header, x_names)
        y_data = self.output_data_splitter.split(used_data, selected_header, y_names)
        # y_data = output_data_cleaner.clean(self.output_data_splitter.split(data, header, Y_NAMES), incorrect_rows)

        x_data = self.transform_variables(x_data, x_names)

        L.info("The used data for the prediction has shape: %s %s" % np.shape(x_data))
        L.info("The values to predict have the shape: %s %s" % np.shape(y_data))

        # Convert ydata 2d matrix (x * 1) to 1d array (x). Needed for the classifcation things
        y_data = np.ravel(y_data)
        return (x_data, y_data, used_data, selected_header)

    def calculate(self, models, x_data, y_data, x_names, y_names):
        # if self.HPC:
            # model_runner = DistributedModelRunner(models)
        # else:
        model_runner = SyncModelRunner(models)

        fabricated_models = model_runner.fabricate_models(x_data, y_data, x_names, y_names, self.VERBOSITY)

        # Train all models, the fitted parameters will be saved inside the models
        return model_runner.run_calculations(fabricated_models=fabricated_models)

    def transform_variables(self, x_data, x_names):
        if self.NORMALIZE:
            L.info('We are also normalizing the features')
            x_data = normalize(x_data)

        if self.SCALE:
            L.info('We are also scaling the features')
            x_data = scale(x_data)

        if self.POLYNOMIAL_FEATURES:
            L.info('We are also adding polynomial features')
            x_data = self.data_preprocessor_polynomial.process(x_data, x_names)

        # Logtransform the data
        # self.variable_transformer.log_transform(x_data, 'aids-somScore')
        return x_data

    def create_output(self, models, y_data, used_data, selected_header, model_type='classification'):
        if model_type == 'classification':
            L.info('In the output set, %d participants (%0.2f percent) is true' %
                   self.calculate_true_false_ratio(y_data))

            # Generate learning curve plots
            self.roc_curve_plotter.plot(models)

        # Export all used data to a CSV file
        CsvExporter.export('../exports/merged_dataframe.csv', used_data, selected_header)

        for model in models:
            1
            # learning_curve_plotter.plot(model)
            # validation_curve_plotter.plot(model, variable_to_validate='n_estimators')

        for model in models:
            model.print_evaluation()
            y_train_pred = model.skmodel.predict(model.x_train)
            y_test_pred = model.skmodel.predict(model.x_test)

            if model_type == 'classification':
                self.confusion_matrix_plotter.plot(model, model.y_test, y_test_pred)
            else:
                self.actual_vs_prediction_plotter.plot_both(model, model.y_test, y_test_pred, model.y_train,
                                                            y_train_pred)

    def create_participants(self):
        data = self.spss_reader.read_file("N1_A100R.sav")
        participants = {}
        for index, entry in data.iterrows():
            p = participant.Participant(entry['pident'], entry['Sexe'], entry['Age'])
            participants[p.pident] = p

        return participants

    def read_cache(self, cache_name):
        with open(cache_name, 'rb') as input:
            header = pickle.load(input)
            data = pickle.load(input)
            return (header, data)

    def write_cache(self, header, data, cache_name):
        with open(cache_name, 'wb') as output:
            pickle.dump(header, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

    def print_header(self, header):
        L.info('Available headers:')
        for col in header:
            L.info('\t' + col)
        L.br()

    def get_file_data(self, file_name, participants, force_to_not_use_cache=False):
        header, data = (None, None)
        L.info('Converting data to single dataframe...')
        if not force_to_not_use_cache and os.path.isfile(file_name):
            header, data = self.read_cache(file_name)
            # self.print_header(header)
        else:
            questionnaires = QuestionnaireFactory.construct_questionnaires(self.spss_reader)
            data, header = (self.single_output_frame_creator.create_single_frame(questionnaires, participants))
            self.write_cache(header, data, file_name)
        return (header, data)

    def calculate_true_false_ratio(self, y_data):
        trues = 0
        falses = 0
        for i in y_data:
            if i == 0:
                falses += 1
            if i == 1:
                trues += 1

        return (trues, (trues / (trues + falses)) * 100)

    def construct_x_names(self):
        """
          Here we select the variables to use in the prediction. The format is:
          AB-C:
          - A = the time of the measurement, a = intake, c = followup
          - B = the name of the questionnaire (check QuestionnaireFactory for the correct names)
          - C = the name of the variable. Check the name used in the <Questionnairename>questionnaire.py
        """
        return np.array([  # 'pident',
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
            # 'a4dkl-somScore',
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
