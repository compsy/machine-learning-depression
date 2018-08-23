import os
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from imblearn.combine import SMOTEENN

from learner.caching.object_cacher import ObjectCacher
from learner.caching.cacher import Cacher
from learner.caching.s3_cacher import S3Cacher
from learner.data_input.spss_reader import SpssReader
from learner.data_output.csv_exporter import CsvExporter
from learner.data_output.datatool_output import DatatoolOutput
from learner.data_output.final_output_generator import OutputGenerator
from learner.data_output.std_logger import L
from learner.data_transformers.data_preprocessor_polynomial import DataPreprocessorPolynomial
from learner.data_transformers.data_resampler import DataResampler
from learner.data_transformers.one_hot_encoder import OneHotEncoderTransformer
from learner.data_transformers.data_scaler import ScalingTransformer
from learner.data_transformers.output_data_cleaner import OutputDataCleaner
from learner.data_transformers.output_data_splitter import OutputDataSplitter
from learner.factories.questionnaire_factory import QuestionnaireFactory
from learner.machine_learning_models.feature_selector import FeatureSelector
from learner.machine_learning_models.model_runners.sync_model_runner import SyncModelRunner
from learner.machine_learning_models.models.boosting_model import BoostingClassificationModel
from learner.machine_learning_models.models.dummy_model import DummyClassifierModel, DummyRandomClassifierModel
from learner.machine_learning_models.models.forest_model import RandomForestClassificationModel
from learner.machine_learning_models.models.neural_network_model import NeuralNetworkModel
from learner.machine_learning_models.models.naive_bayes_model import GaussianNaiveBayesModel, BernoulliNaiveBayesModel
from learner.machine_learning_models.models.regression_model import LogisticRegressionModel
from learner.machine_learning_models.models.stochastic_gradient_descent_model import \
    StochasticGradientDescentClassificationModel
from learner.machine_learning_models.models.support_vector_machine_model import SupportVectorClassificationModel
from learner.machine_learning_models.models.tree_model import ClassificationTreeModel
from learner.models import participant
from learner.output_file_creators.descriptives_table_creator import DescriptivesTableCreator
from learner.output_file_creators.single_output_frame_creator import SingleOutputFrameCreator


class Driver:
    """ The main class that runs the application
    Parameters
    ----------
    verbosity : int, default=0
        A number representing the verbosity of the model calculation.
        1 is on, 0 is off.
    hpc : boolean, default=False
        Should the application run each of the models on a separate node?
        This should not be combined with distributed gridsearch. Also consider
        the logging is done on an HPC? If so, the logging of the slaves is turned off
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
    feature_selection : perform feature selection using elasticnet
    """

    def __init__(self, verbosity, polynomial_features, normalize, scale, force_no_caching, feature_selection, categorical_features_limit=10):
        # Set a seed for reproducability
        # random.seed(42)

        # setup logging
        L.setup(False)
        if (os.environ.get('AWS_CONFIG_FILE') == None):
            L.warn('No AWS config location set! using the default!!!')

        # Define global variables
        self.CATEGORICAL_FEATURES_LIMIT = categorical_features_limit
        self.VERBOSITY = verbosity
        self.POLYNOMIAL_FEATURES = polynomial_features
        self.NORMALIZE = normalize
        self.SCALE = scale
        self.FORCE_NO_CACHING = force_no_caching
        self.FEATURE_SELECTION = feature_selection
        self.TOP = 20

        # create several objects to do data processing
        self.spss_reader = SpssReader()
        self.single_output_frame_creator = SingleOutputFrameCreator()
        self.output_data_splitter = OutputDataSplitter()
        self.cacher = ObjectCacher(directory='cache/')

        self.final_output_generator = OutputGenerator()

        self.feature_selector = FeatureSelector()

        # Input columns
        # Retrieve the names of the variables to use in the prediction
        self.x_names = QuestionnaireFactory.construct_x_names()

        # Output columns
        # self.classification_y_names = np.array(['ccidi-depression-followup-majorDepressionPastSixMonths'])
        self.classification_y_names = np.array(['cids-followup-twice_depression'])

        ##### Define the models we should run
        self.classification_models = []
        self.classification_models.append({'model': ClassificationTreeModel, 'options': ['grid-search']})
        self.classification_models.append({'model': StochasticGradientDescentClassificationModel, 'options': ['grid-search']})
        self.classification_models.append({'model': RandomForestClassificationModel, 'options': ['grid-search']})
        self.classification_models.append({'model': DummyClassifierModel, 'options': []})
        self.classification_models.append({'model': DummyRandomClassifierModel, 'options': []})
        self.classification_models.append({'model': SupportVectorClassificationModel, 'options': ['grid-search']})
        self.classification_models.append({'model': BoostingClassificationModel, 'options': ['grid-search']})
        self.classification_models.append({'model': LogisticRegressionModel, 'options': ['grid-search']})
        self.classification_models.append({'model': BernoulliNaiveBayesModel, 'options': ['grid-search']})
        # self.classification_models.append({'model': NeuralNetworkModel, 'options': ['grid-search']})

        # self.classification_models.append({'model': StochasticGradientDescentClassificationModel, 'options': ['bagging']})
        # self.classification_models.append({'model': RandomForestClassificationModel, 'options': ['bagging']})
        # self.classification_models.append({'model': DummyClassifierModel, 'options': ['bagging']})
        # self.classification_models.append({'model': DummyRandomClassifierModel, 'options': ['bagging']})
        # self.classification_models.append({'model': ClassificationTreeModel, 'options': ['bagging']})
        # # self.classification_models.append({'model': SupportVectorClassificationModel, 'options': ['bagging']})
        # self.classification_models.append({'model': BoostingClassificationModel, 'options': ['bagging']})
        # self.classification_models.append({'model': LogisticRegressionModel, 'options': ['bagging']})
        # self.classification_models.append({'model': BernoulliNaiveBayesModel, 'options': ['bagging']})

        # regression_models = []
        # regression_models.append({'model': ElasticNetModel, 'options':[]})
        # regression_models.append({'model': SupportVectorRegressionModel, 'options':[]})
        # regression_models.append({'model': RegressionTreeModel, 'options': []})
        # regression_models.append(BoostingModel)
        # regression_models.append(BaggingModel)

    def run_setcreator(self, test_set_portion=0.2):
        """
        The main function that is called when creating the sets
        Parameters
        ----------
        test_set_portion : double, default=0.2 the portion of the data that should be used for testing
        """
        start_time = time.monotonic()
        # Delete all cached models
        S3Cacher(directory='cache', preload=False).delete_all()
        Cacher.clean_cache('cache/mlmodels/')
        DatatoolOutput.clear()

        test_set_id = 1
        participants = self.create_participants()
        DatatoolOutput.export('total-number-of-participants', len(participants))

        data = self.get_file_data('cache', participants=participants, force_to_not_use_cache=self.FORCE_NO_CACHING)


        # L.info('We have %d participants in the inital dataset' % len(participants.keys()))

        #### Classification ####
        CsvExporter.export('exports/merged_all_dataframe.csv', data)

        L.info('We are using %s as input.' % self.x_names)

        DatatoolOutput.export('total-percentage-female', round(sum([p.gender for k, p in participants.items()]) / len(participants) * 100, 3))

        # We use the last column to get an impression of this number. According to the ducmentation it should be 2596
        n_followup_participants = np.shape(data)[0] - np.sum(np.isnan(data[data.columns[-1]]))
        if(n_followup_participants != 2596):
            throw('The number of follow up participants is incorrect!')

        DatatoolOutput.export('total-number-of-participants-followup', n_followup_participants)
        DatatoolOutput.export('total-number-of-participants-followup-percentage', round((n_followup_participants / len(participants)) * 100, 3))

        # Clean the data, remove the incomplete cases
        x_data, y_data = self.get_usable_data(data, self.x_names, self.classification_y_names)

        coefficients = None
        DatatoolOutput.export('total-number-of-features', len(self.x_names))
        DatatoolOutput.export('total-number-of-used-participants-followup-percentage', round((len(self.x_names) / n_followup_participants) * 100, 3))

        # Perform feature selection algorithm
        if self.FEATURE_SELECTION:
            coefficients = self.perform_feature_selection(
                x_data, y_data, self.classification_y_names, model_type='classification')

            # Update the x_names according to the found coefficients (this performs the actual feature selection)
            self.x_names = coefficients[0:, 0]
            x_data = x_data[self.x_names]


        nobs = x_data.shape[0]
        split_set = np.random.binomial(test_set_id, test_set_portion, nobs)

        x_data = x_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)
        DatatoolOutput.export('outcome-is-one-percentage', round(100 * (pd.DataFrame.sum(y_data)[0]/len(y_data)), 3))
        DatatoolOutput.export('outcome-is-one-percentage-rounded', math.ceil(100 * (pd.DataFrame.sum(y_data)[0]/len(y_data))))
        DatatoolOutput.export('total-size-row', np.shape(x_data)[0])

        x_data['test'] = split_set
        y_data['test'] = split_set

        DescriptivesTableCreator.generate_coefficient_descriptives_table(
            x_data, coefficients, name='z_classification_descriptives')

        # Create test and training set
        x_data_train = x_data.loc[x_data['test'] != test_set_id].drop('test', 1).reset_index(drop=True)
        y_data_train = y_data.loc[y_data['test'] != test_set_id].drop('test', 1).reset_index(drop=True)
        DatatoolOutput.export('train-size-row', np.shape(x_data_train)[0])
        DatatoolOutput.export('outcome-is-one-percentage-train', round(100 * (pd.DataFrame.sum(y_data_train)[0]/len(y_data_train)), 3))

        # resampling is needed because the higlhy unbalanced datasets
        x_data_train, y_data_train = DataResampler.process(x_data = x_data_train, y_data = y_data_train)
        DatatoolOutput.export('train-size-row-after-resampling', np.shape(x_data_train)[0])
        DatatoolOutput.export('outcome-is-one-percentage-train-after-resampling', round(100 * (pd.DataFrame.sum(y_data_train)[0]/len(y_data_train)), 3))

        training_data = {
            'x_data': x_data_train,
            'y_data': y_data_train
        }
        self.cacher.write_cache(training_data, 'training_data.pkl')
        CsvExporter.export('exports/merged_train_dataframe.csv', pd.concat([x_data_train, y_data_train], axis = 1))

        x_data_test = x_data.loc[x_data['test'] == test_set_id].drop('test', 1).reset_index(drop=True)
        y_data_test = y_data.loc[y_data['test'] == test_set_id].drop('test', 1).reset_index(drop=True)
        DatatoolOutput.export('test-size-row', np.shape(x_data_test)[0])
        DatatoolOutput.export('outcome-is-one-percentage-test', round(100 * (pd.DataFrame.sum(y_data_test)[0]/len(y_data_test)), 3))

        # x_data_test, y_data_test = DataResampler.process(x_data = x_data_test, y_data = y_data_test)
        test_data = {
            'x_data': x_data_test,
            'y_data': y_data_test,
            'all_data': pd.concat([x_data, y_data], axis=1)
        }
        self.cacher.write_cache(test_data, 'test_data.pkl')
        CsvExporter.export('exports/merged_test_dataframe.csv', pd.concat([x_data_test, y_data_test], axis = 1))
        CsvExporter.export('exports/merged_full_pre_dataframe.csv', pd.concat([x_data, y_data], axis = 1))
        end_time = time.monotonic()
        self.export_time('evaluation', start = start_time, end = end_time)


    def run_evaluator(self):
        """
        Main method to run the evaluation on all the machine learning algorithms fitted earlier
        """
        start_time = time.monotonic()
        x_data, y_data, all_data = self.load_data(filename='test_data.pkl')
        L.info('Running evaluation on data set of size (%d, %d)' % np.shape(x_data))
        L.info('The headers in this file are: %s' % x_data.columns)

        # Calculate the actual models
        model_runner = SyncModelRunner(self.classification_models)
        fabricate_models = model_runner.fabricate_models(
            x_data, y_data, self.classification_y_names, verbosity=self.VERBOSITY)

        self.final_output_generator.create_output(
            fabricate_models=fabricate_models,
            x_data=x_data,
            y_data=y_data,
            output_type='test',
            model_type='classification')

        # Export all used data to a CSV file
        CsvExporter.export('exports/merged_full_dataframe.csv', all_data)
        end_time = time.monotonic()
        self.export_time('evaluation', start = start_time, end = end_time)

    def run_trainer(self):
        """
        Main method to train all the machine learning algorithms
        """
        start_time = time.monotonic()
        if not self.cacher.file_available('training_data.pkl', add_dir=True):
            raise FileNotFoundError('Training data not found!')

        # Retrieve the data from the cache
        x_data, y_data = self.load_data(filename='training_data.pkl')

        DatatoolOutput.export('number-of-ml-algorithms', DatatoolOutput.number_to_string(len(self.classification_models)))
        # Calculate the actual models
        model_runner = SyncModelRunner(self.classification_models)

        fabricated_models = model_runner.fabricate_models(
            x_data, y_data, self.classification_y_names, verbosity=self.VERBOSITY)

        model_runner.run_calculations(fabricated_models=fabricated_models)
        end_time = time.monotonic()
        self.export_time('training', start = start_time, end = end_time)


    def perform_feature_selection(self, x_data, y_data, y_names, model_type):
        """
        Performs feature selection based on the elasticnet
        """
        temp_pol_features = self.POLYNOMIAL_FEATURES
        self.POLYNOMIAL_FEATURES = False

        L.info('Performing feature selection for ' + model_type)

        feature_selection_model = StochasticGradientDescentClassificationModel(
            x_data, y_data, y_names, grid_search=False, verbosity=0)

        feature_selection_model.train(cache_result=False)

        DatatoolOutput.export('number-of-features', self.TOP)
        coefficients = self.feature_selector.determine_best_variables(feature_selection_model, top = self.TOP)
        self.POLYNOMIAL_FEATURES = temp_pol_features
        return coefficients

    def export_time(self, key, start, end):
        delta = end - start
        DatatoolOutput.export(key+'-time-seconds', round(delta, 3))
        DatatoolOutput.export(key+'-time-minutes', round(delta / 60, 3))
        DatatoolOutput.export(key+'-time-hours', round(delta / 60 / 60, 3))


    def load_data(self, filename):
        """
        Loads the training or testing data from a file provided to it
        """
        data = self.cacher.read_cache(filename)

        x_data = data['x_data']
        y_data = data['y_data']
        if 'all_data' in data:
            all_data = data['all_data']
            return (x_data, y_data, all_data)
        return (x_data, y_data)

    def get_usable_data(self, data, x_names, y_names):
        L.info('Loaded data with %d rows and %d columns' % np.shape(data))
        L.info('We will use %s as outcome.' % y_names)

        selected_header = np.append(x_names, y_names)

        # Select the data we will use in the present experiment (used_data = both x and y)
        used_data = data[selected_header]

        # Determine which of this set are not complete
        incorrect_rows = OutputDataCleaner.find_incomplete_rows(used_data, selected_header)

        L.info('From the loaded data %d rows are incomplete and will be removed!' % len(incorrect_rows))

        # Remove the incorrect cases
        used_data = OutputDataCleaner.clean(used_data, incorrect_rows)

        # Split the dataframe into a x and y dataset.
        x_data = used_data[x_names]
        y_data = used_data[y_names]

        # y_data = output_data_cleaner.clean(self.output_data_splitter.split(data, header, Y_NAMES), incorrect_rows)

        x_data = self.transform_variables(x_data)
        if np.shape(x_data)[1] is not len(x_names):
            L.warn('The dimension of the X names data is not equal to the dimension of the data')
            L.warn('The dimensions are: %s and %d' % (np.shape(x_data)[1], len(x_names)))

        L.info("The used data for the prediction has shape: %s %s" % np.shape(x_data))
        L.info("The values to predict have the shape: %s %s" % np.shape(y_data))

        # Convert ydata 2d matrix (x * 1) to 1d array (x). Needed for the classifcation things
        return (x_data, y_data)

    def transform_variables(self, x_data):
        """
        Transforms variables to include different features
        """

        # Save the names (they will be removed by scikitlearn)
        names = x_data.columns

        if self.CATEGORICAL_FEATURES_LIMIT > 0:
            x_data = OneHotEncoderTransformer.perform_encoding(x_data, limit=self.CATEGORICAL_FEATURES_LIMIT)
            names = x_data.columns

        if self.NORMALIZE:
            L.info('We are also normalizing the features')
            np_x_data = normalize(x_data, norm='l2', axis=1)
            x_data = pd.DataFrame(np_x_data, columns=names)

        if self.POLYNOMIAL_FEATURES:
            L.info('We are also adding polynomial features')
            # We don't have to poor the data into a dataframe here, as the processor does it for us
            x_data = DataPreprocessorPolynomial.process(x_data)
            names = x_data.columns

        if self.SCALE:
            L.info('We are also scaling the features')
            ScalingTransformer.perform_scaling(x_data, scale_binary=False)

        # Logtransform the data
        # variable_transformer = VariableTransformer(self.x_names)
        # variable_transformer.log_transform(x_data, 'aids-somScore')
        return x_data

    def create_participants(self, participant_file="N1_A100R.sav"):
        data = self.spss_reader.read_file(participant_file)
        participants = {}
        for index, entry in data.iterrows():
            p = participant.Participant(entry['pident'], entry['Sexe'], entry['Age'])
            participants[p.pident] = p

        return participants

    def print_header(self, header):
        L.info('Available headers:')
        for col in header:
            L.info('\t' + col)
        L.br()

    def get_file_data(self, file_name, participants, force_to_not_use_cache=False):
        data = None
        data_file_name = file_name + '_data.pkl'
        L.info('Converting data to single dataframe...')
        if not force_to_not_use_cache and self.cacher.file_available(data_file_name):
            data = self.cacher.read_cache(data_file_name)
        else:
            questionnaires = QuestionnaireFactory.construct_questionnaires(self.spss_reader)
            data = (self.single_output_frame_creator.create_single_frame(questionnaires, participants))
            self.cacher.write_cache(data, data_file_name)

        return data
