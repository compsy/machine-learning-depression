import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, scale
import matplotlib.pyplot as plt
import os

from learner.caching.object_cacher import ObjectCacher
from learner.data_input.spss_reader import SpssReader
from learner.data_output.csv_exporter import CsvExporter
from learner.data_output.final_output_generator import OutputGenerator
from learner.data_output.std_logger import L
from learner.data_transformers.data_preprocessor_polynomial import DataPreprocessorPolynomial
from learner.data_transformers.output_data_cleaner import OutputDataCleaner
from learner.data_transformers.output_data_splitter import OutputDataSplitter
from learner.data_transformers.variable_transformer import VariableTransformer
from learner.factories.questionnaire_factory import QuestionnaireFactory
from learner.machine_learning_models.feature_selector import FeatureSelector
from learner.machine_learning_models.model_runners.sync_model_runner import SyncModelRunner
from learner.machine_learning_models.models.boosting_model import BoostingClassificationModel
from learner.machine_learning_models.models.dummy_model import DummyClassifierModel, DummyRandomClassifierModel
from learner.machine_learning_models.models.forest_model import RandomForestClassificationModel
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

    def __init__(self, verbosity, polynomial_features, normalize, scale, force_no_caching, feature_selection):
        # Set a seed for reproducability
        # random.seed(42)

        # setup logging
        L.setup(False)
        if(os.environ.get('AWS_CONFIG_FILE') == None):
            L.warn('No AWS config location set! using the default!!!')

        # Define global variables
        self.VERBOSITY = verbosity
        self.POLYNOMIAL_FEATURES = polynomial_features
        self.NORMALIZE = normalize
        self.SCALE = scale
        self.FORCE_NO_CACHING = force_no_caching
        self.FEATURE_SELECTION = feature_selection

        # create several objects to do data processing
        self.spss_reader = SpssReader()
        self.single_output_frame_creator = SingleOutputFrameCreator()
        self.output_data_cleaner = OutputDataCleaner()
        self.output_data_splitter = OutputDataSplitter()
        self.data_preprocessor_polynomial = DataPreprocessorPolynomial()
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
        # self.classification_models.append(KerasNnClassificationModel)
        self.classification_models.append({'model': ClassificationTreeModel, 'options': ['grid-search']})
        self.classification_models.append({'model': StochasticGradientDescentClassificationModel, 'options': ['grid-search']})
        self.classification_models.append({'model': RandomForestClassificationModel, 'options': ['grid-search']})
        self.classification_models.append({'model': DummyClassifierModel, 'options': []})
        self.classification_models.append({'model': DummyRandomClassifierModel, 'options': []})
        self.classification_models.append({'model': SupportVectorClassificationModel, 'options': ['grid-search']})
        self.classification_models.append({'model': BoostingClassificationModel, 'options': ['grid-search']})
        self.classification_models.append({'model': LogisticRegressionModel, 'options': ['grid-search']})
        self.classification_models.append({'model': GaussianNaiveBayesModel, 'options': ['grid-search']})
        self.classification_models.append({'model': BernoulliNaiveBayesModel, 'options': ['grid-search']})

        self.classification_models.append({'model': StochasticGradientDescentClassificationModel, 'options': ['bagging']})
        self.classification_models.append({'model': RandomForestClassificationModel, 'options': ['bagging']})
        self.classification_models.append({'model': DummyClassifierModel, 'options': ['bagging']})
        self.classification_models.append({'model': DummyRandomClassifierModel, 'options': ['bagging']})
        self.classification_models.append({'model': ClassificationTreeModel, 'options': ['bagging']})
        self.classification_models.append({'model': SupportVectorClassificationModel, 'options': ['bagging']})
        self.classification_models.append({'model': BoostingClassificationModel, 'options': ['bagging']})
        self.classification_models.append({'model': LogisticRegressionModel, 'options': ['bagging']})
        self.classification_models.append({'model': GaussianNaiveBayesModel, 'options': ['bagging']})
        self.classification_models.append({'model': BernoulliNaiveBayesModel, 'options': ['bagging']})

        # regression_models = []
        # regressionmodels.append(KerasNnModel)
        # regression_models.append({'model': ElasticNetModel, 'options':[]})
        # regression_models.append({'model': SupportVectorRegressionModel, 'options':[]})
        # regression_models.append({'model': RegressionTreeModel, 'options': []})
        # regression_models.append(BoostingModel)
        # regression_models.append(BaggingModel)

    def run_setcreator(self, test_set_portion = 0.2):
        test_set_id = 1
        participants = self.create_participants()
        data = self.get_file_data(
            'cache', participants=participants, force_to_not_use_cache=self.FORCE_NO_CACHING)

        # L.info('We have %d participants in the inital dataset' % len(participants.keys()))

        #### Classification ####
        # Perform feature selection algorithm
        CsvExporter.export('exports/merged_all_dataframe.csv', data)

        coefficients = None
        if (self.FEATURE_SELECTION):
            coefficients = self.perform_feature_selection(
                data, self.x_names, self.classification_y_names, model_type='classification')
            self.x_names = coefficients[0:, 0]

        L.info('We are using %s as input.' % self.x_names)
        x_data, y_data, used_data, selected_header = self.get_usable_data(
                data, self.x_names, self.classification_y_names)


        n = x_data.shape[0]
        split_set =  np.random.binomial(test_set_id, test_set_portion, n)
        x_data['test'] = split_set
        y_data['test'] = split_set

        DescriptivesTableCreator.generate_coefficient_descriptives_table(
            x_data, coefficients, name='classification_descriptives')

        self.variable_transformer = VariableTransformer(self.x_names)

        # Create test and training set
        training_data = {
            'x_data': x_data.loc[x_data['test'] != test_set_id].drop('test', 1),
            'y_data': y_data.loc[y_data['test'] != test_set_id].drop('test', 1),
            'used_data': used_data,
            'selected_header': selected_header
        }

        test_data = {
            'x_data': x_data.loc[x_data['test'] == test_set_id].drop('test', 1),
            'y_data': y_data.loc[y_data['test'] == test_set_id].drop('test', 1),
            'used_data': used_data,
            'selected_header': selected_header
        }

        self.cacher.write_cache(training_data, 'training_data.pkl')
        self.cacher.write_cache(test_data, 'test_data.pkl')

    def run_evaluator(self):
        x_data, y_data, used_data, selected_header = self.load_data(filename = 'test_data.pkl')

        # Calculate the actual models
        model_runner = SyncModelRunner(self.classification_models)
        classification_fabricated_models = model_runner.fabricate_models(x_data, y_data, self.x_names, self.classification_y_names, verbosity=self.VERBOSITY)

        self.final_output_generator.create_output(
            classification_fabricated_models,
            y_data,
            used_data,
            selected_header,
            output_type = 'test',
            model_type='classification')

    def run_trainer(self):
        if not self.cacher.file_available('training_data.pkl', add_dir=True):
            raise FileNotFoundError('Training data not found!')
        x_data, y_data, used_data, selected_header = self.load_data(filename = 'training_data.pkl')

        # Calculate the actual models
        model_runner = SyncModelRunner(self.classification_models)

        classification_fabricated_models = model_runner.fabricate_models(x_data, y_data, self.x_names, self.classification_y_names, verbosity=self.VERBOSITY)

        # TODO: ??? SHOULD WE STILL DO THIS? Train all models, the fitted parameters will be saved inside the models
        model_runner.run_calculations(fabricated_models=classification_fabricated_models)

    def perform_feature_selection(self, data, x_names, y_names, model_type):
        temp_pol_features = self.POLYNOMIAL_FEATURES
        self.POLYNOMIAL_FEATURES = False
        usable_x_data, usable_y_data, used_data, selected_header = self.get_usable_data(data, x_names, y_names)
        L.info('Performing feature selection for ' + model_type)

        feature_selection_model = StochasticGradientDescentClassificationModel(
            np.copy(usable_x_data), np.copy(usable_y_data), x_names, y_names, grid_search=False, verbosity=0
            )
        feature_selection_model.train(cache_result=False)
        coefficients = self.feature_selector.determine_best_variables(feature_selection_model)
        self.POLYNOMIAL_FEATURES = temp_pol_features
        return coefficients

    def load_data(self, filename):
        training_data = self.cacher.read_cache(filename)

        x_data = training_data['x_data']
        y_data = training_data['y_data']
        used_data = training_data['used_data']
        selected_header = training_data['selected_header']
        return(x_data, y_data, used_data, selected_header)


    def get_usable_data(self, data, x_names, y_names):
        L.info('Loaded data with %d rows and %d columns' % np.shape(data))
        L.info('We will use %s as outcome.' % y_names)

        selected_header = np.append(x_names, y_names)

        # Select the data we will use in the present experiment (used_data = both x and y)
        used_data = data[selected_header]

        # Determine which of this set are not complete
        # import pdb
        # pdb.set_trace()
        incorrect_rows = self.output_data_cleaner.find_incomplete_rows(used_data, selected_header)

        L.info('From the loaded data %d rows are incomplete and will be removed!' % len(incorrect_rows))

        # Remove the incorrect cases
        used_data = self.output_data_cleaner.clean(used_data, incorrect_rows)

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
        return (x_data, y_data, used_data, selected_header)

    def transform_variables(self, x_data):
        # Save the names (they will be removed by scikitlearn)
        names = list(x_data)

        if self.NORMALIZE:
            L.info('We are also normalizing the features')
            x_data = normalize(x_data, norm='l2', axis=1)
            x_data = pd.DataFrame(x_data, columns=names)


        if self.SCALE:
            L.info('We are also scaling the features')
            x_data = scale(x_data)
            x_data = pd.DataFrame(x_data, columns=names)

        if self.POLYNOMIAL_FEATURES:
            L.info('We are also adding polynomial features')
            x_data = self.data_preprocessor_polynomial.process(x_data)

        # Logtransform the data
        # self.variable_transformer.log_transform(x_data, 'aids-somScore')
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
        header, data = (None, None)
        data_file_name = file_name + '_data.pkl'
        L.info('Converting data to single dataframe...')
        if not force_to_not_use_cache and self.cacher.file_available(data_file_name):
            data = self.cacher.read_cache(data_file_name)
        else:
            questionnaires = QuestionnaireFactory.construct_questionnaires(self.spss_reader)
            data = (self.single_output_frame_creator.create_single_frame(questionnaires, participants))
            self.cacher.write_cache(data, data_file_name)

        return data
