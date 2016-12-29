import os.path
import random

import numpy as np
from mpi4py import MPI
from sklearn.preprocessing import normalize, scale

from learner.caching.object_cacher import ObjectCacher
from learner.data_input.spss_reader import SpssReader
from learner.data_output.csv_exporter import CsvExporter
from learner.data_output.latex_table_exporter import LatexTableExporter
from learner.data_output.plotters.actual_vs_prediction_plotter import ActualVsPredictionPlotter
from learner.data_output.plotters.confusion_matrix_plotter import ConfusionMatrixPlotter
from learner.data_output.plotters.data_density_plotter import DataDensityPlotter
from learner.data_output.plotters.learning_curve_plotter import LearningCurvePlotter
from learner.data_output.plotters.roc_curve_plotter import RocCurvePlotter
from learner.data_output.plotters.validation_curve_plotter import ValidationCurvePlotter
from learner.data_output.std_logger import L
from learner.data_transformers.data_preprocessor_polynomial import DataPreprocessorPolynomial
from learner.data_transformers.output_data_cleaner import OutputDataCleaner
from learner.data_transformers.output_data_splitter import OutputDataSplitter
from learner.data_transformers.variable_transformer import VariableTransformer
from learner.factories.questionnaire_factory import QuestionnaireFactory
from learner.machine_learning_evaluation.true_false_ratio_evaluation import TrueFalseRationEvaluation
from learner.machine_learning_models.feature_selector import FeatureSelector
from learner.machine_learning_models.models.boosting_model import BoostingClassificationModel
from learner.machine_learning_models.models.dummy_model import DummyClassifierModel, DummyRandomClassifierModel
from learner.machine_learning_models.models.forest_model import RandomForestClassificationModel
from learner.machine_learning_models.models.naive_bayes_model import NaiveBayesModel
from learner.machine_learning_models.models.regression_model import ElasticNetModel, LogisticRegressionModel
from learner.machine_learning_models.models.support_vector_machine_model import SupportVectorRegressionModel, \
        SupportVectorClassificationModel
from learner.machine_learning_models.models.tree_model import RegressionTreeModel, ClassificationTreeModel
from learner.machine_learning_models.model_runners.sync_model_runner import SyncModelRunner
from learner.models import participant
from learner.output_file_creators.descriptives_table_creator import DescriptivesTableCreator
from learner.output_file_creators.single_output_frame_creator import SingleOutputFrameCreator
from learner.machine_learning_models.models.naive_bayes_model import GaussianNaiveBayesModel, BernoulliNaiveBayesModel
from learner.machine_learning_models.models.stochastic_gradient_descent_model import \
    StochasticGradientDescentClassificationModel


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

    def __init__(self, verbosity, hpc, polynomial_features, normalize, scale, force_no_caching, feature_selection):
        # Set a seed for reproducability
        # random.seed(42)
        self.comm = MPI.COMM_WORLD

        if hpc:
            print('[HPC] Node %d initialized.' % self.comm.Get_rank())

        # setup logging
        L.setup(hpc)

        # Define global variables
        self.VERBOSITY = verbosity
        self.HPC = hpc
        self.POLYNOMIAL_FEATURES = polynomial_features
        self.NORMALIZE = normalize
        self.SCALE = scale
        self.FORCE_NO_CACHING = force_no_caching
        self.FEATURE_SELECTION = feature_selection

        # # Create objects to perform image plotting
        self.actual_vs_prediction_plotter = ActualVsPredictionPlotter()
        self.learning_curve_plotter = LearningCurvePlotter()
        self.validation_curve_plotter = ValidationCurvePlotter()
        self.roc_curve_plotter = RocCurvePlotter()
        self.confusion_matrix_plotter = ConfusionMatrixPlotter()
        self.data_density_plotter = DataDensityPlotter()

        # create several objects to do data processing
        self.spss_reader = SpssReader()
        self.single_output_frame_creator = SingleOutputFrameCreator()
        self.output_data_cleaner = OutputDataCleaner()
        self.output_data_splitter = OutputDataSplitter()
        self.data_preprocessor_polynomial = DataPreprocessorPolynomial()
        self.cacher = ObjectCacher()

        self.feature_selector = FeatureSelector()

    def run(self):
        # Retrieve the names of the variables to use in the prediction
        x_names = QuestionnaireFactory.construct_x_names()

        ##### Define the models we should run
        classification_models = []
        # classification_models.append(KerasNnClassificationModel)
        classification_models.append({'model': ClassificationTreeModel, 'options': ['grid-search']})
        #classification_models.append({'model': StochasticGradientDescentClassificationModel, 'options': ['grid-search']})
        #classification_models.append({'model': RandomForestClassificationModel, 'options': ['grid-search']})
        classification_models.append({'model': DummyClassifierModel, 'options': []})
        classification_models.append({'model': DummyRandomClassifierModel, 'options': []})
        # classification_models.append({'model': SupportVectorClassificationModel, 'options': ['grid-search']})
        #classification_models.append({'model': BoostingClassificationModel, 'options': ['grid-search']})
        #classification_models.append({'model': LogisticRegressionModel, 'options': ['grid-search']})
        classification_models.append({'model': GaussianNaiveBayesModel, 'options': ['grid-search']})
        #classification_models.append({'model': BernoulliNaiveBayesModel, 'options': ['grid-search']})

        #classification_models.append({'model': StochasticGradientDescentClassificationModel, 'options': ['bagging']})
        # classification_models.append({'model': RandomForestClassificationModel, 'options': ['bagging']})
        classification_models.append({'model': DummyClassifierModel, 'options': ['bagging']})
        classification_models.append({'model': DummyRandomClassifierModel, 'options': ['bagging']})
        classification_models.append({'model': ClassificationTreeModel, 'options': ['bagging']})
        #classification_models.append({'model': SupportVectorClassificationModel, 'options': ['bagging']})
        #classification_models.append({'model': BoostingClassificationModel, 'options': ['bagging']})
        #classification_models.append({'model': LogisticRegressionModel, 'options': ['bagging']})
        classification_models.append({'model': GaussianNaiveBayesModel, 'options': ['bagging']})
        #classification_models.append({'model': BernoulliNaiveBayesModel, 'options': ['bagging']})

        #regression_models = []
        # regressionmodels.append(KerasNnModel)
        # regression_models.append({'model': ElasticNetModel, 'options':[]})
        # regression_models.append({'model': SupportVectorRegressionModel, 'options':[]})
        #regression_models.append({'model': RegressionTreeModel, 'options': []})
        # regression_models.append(BoostingModel)
        # regression_models.append(BaggingModel)

        # Output columns
        # classification_y_names = np.array(['ccidi-depression-followup-majorDepressionPastSixMonths'])
        classification_y_names = np.array(['cids-followup-twice_depression'])

        #regression_y_names = np.array(['cids-followup-somScore'])
        if self.comm.Get_rank() == 0 or not self.HPC:
            participants = self.create_participants()
            header, data = self.get_file_data(
                'cache', participants=participants, force_to_not_use_cache=self.FORCE_NO_CACHING)

            # L.info('We have %d participants in the inital dataset' % len(participants.keys()))

            #### Classification ####
            # Perform feature selection algorithm
            CsvExporter.export('exports/merged_all_dataframe.csv', data, header)

            coefficients = None
            if (self.FEATURE_SELECTION):
                coefficients = self.perform_feature_selection(
                    data, header, x_names, classification_y_names, model_type='classification')
                x_names = coefficients[0:, 0]

            L.info('We are using %s as input.' % x_names)
            x_data, classification_y_data, used_data, selected_header = self.get_usable_data(data, header, x_names,
                                                                                             classification_y_names)
            DescriptivesTableCreator.generate_coefficient_descriptives_table(
                x_data, x_names, coefficients, name='classification_descriptives')
            self.variable_transformer = VariableTransformer(x_names)
        else:
            x_data=None
            classification_y_data= None
            used_data= None
            selected_header= None

        if self.HPC:
            L.info('[HPC-Master] Sending data to nodes for data from node %d' % self.comm.Get_rank())
            x_data = self.comm.bcast(x_data, root=0)
            classification_y_data = self.comm.bcast(classification_y_data, root=0)
            used_data = self.comm.bcast(used_data, root=0)
            selected_header = self.comm.bcast(selected_header, root=0)

        self.comm.Barrier()
        # Calculate the actual models
        model_runner = SyncModelRunner(classification_models, hpc=self.HPC)
        is_root, classification_fabricated_models = model_runner.calculate(
            x_data, classification_y_data, x_names, classification_y_names, verbosity=self.VERBOSITY)

        ############################################################################################################
        #### Regression ####
        # Reset the names to the original set
        #x_names = QuestionnaireFactory.construct_x_names()
        # Perform feature selection algorithm
        # coefficients = None
        # if (self.FEATURE_SELECTION):
        #     coefficients = self.perform_feature_selection(
        #         data, header, x_names, regression_y_names, model_type='regression')
        #     x_names = coefficients[0:, 0]

        #L.info('We are using %s as input.' % x_names)
        #x_data, regression_y_data, used_data, selected_header = self.get_usable_data(data, header, x_names,
        #                                                                             regression_y_names)

        #DescriptivesTableCreator.generate_coefficient_descriptives_table(
        #    x_data, x_names, coefficients, name='regression_descriptives')
        #self.variable_transformer = VariableTransformer(x_names)

        # Calculate the actual models
        #model_runner = SyncModelRunner(regression_models, hpc=self.HPC)
        #is_root, regression_fabricated_models = model_runner.calculate(
        #    x_data, regression_y_data, x_names, regression_y_names, verbosity=self.VERBOSITY)
        ############################################################################################################

        # Kill all worker nodes
        if self.HPC and MPI.COMM_WORLD.Get_rank() > 0:
            L.info('Byebye from node %d' % MPI.COMM_WORLD.Get_rank(), force=True)
            exit(0)

        # Plot an overview of the density estimations of the variables used in the actual model calculation.
        #DescriptivesTableCreator.create_data_descriptive_plots(participants, x_data, x_names)
        self.create_output(
            classification_fabricated_models,
            classification_y_data,
            used_data,
            selected_header,
            model_type='classification')
        #self.create_output(
        #    regression_fabricated_models, regression_y_data, used_data, selected_header, model_type='regression')

    def perform_feature_selection(self, data, header, x_names, y_names, model_type):
        temp_pol_features = self.POLYNOMIAL_FEATURES
        self.POLYNOMIAL_FEATURES = False
        usable_x_data, usable_y_data, used_data, selected_header = self.get_usable_data(data, header, x_names, y_names)
        L.info('Performing feature selection for ' + model_type)

        feature_selection_model = StochasticGradientDescentClassificationModel(
            np.copy(usable_x_data), np.copy(usable_y_data), x_names, y_names, grid_search=False, verbosity=0, hpc=self.HPC)
        feature_selection_model.train()
        coefficients = self.feature_selector.determine_best_variables(feature_selection_model)
        self.POLYNOMIAL_FEATURES = temp_pol_features
        return coefficients

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
        if np.shape(x_data)[1] is not len(x_names):
            L.warn('The dimension of the X names data is not equal to the dimension of the data')
            L.warn('The dimensions are: %s and %d' % (np.shape(x_data)[1], len(x_names)))

        L.info("The used data for the prediction has shape: %s %s" % np.shape(x_data))
        L.info("The values to predict have the shape: %s %s" % np.shape(y_data))

        # Convert ydata 2d matrix (x * 1) to 1d array (x). Needed for the classifcation things
        y_data = np.ravel(y_data)
        return (x_data, y_data, used_data, selected_header)

    def transform_variables(self, x_data, x_names):
        if self.NORMALIZE:
            L.info('We are also normalizing the features')
            x_data = normalize(x_data, norm='l2', axis=1)

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
            true_false_ration_evaluation = TrueFalseRationEvaluation(pos_label=0)
            train_trues, train_outcome, test_trues, test_outcome = true_false_ration_evaluation\
                .evaluate(y_train=y_data, y_test=models[0].y_test)
            L.info('In the training set, %d participants (%0.2f percent) is true' % (train_trues, train_outcome))
            L.info('In the test set, %d participants (%0.2f percent) is true' % (test_trues, test_outcome))

            # Generate learning curve plots
            self.roc_curve_plotter.plot(models)

        # Export all used data to a CSV file
        CsvExporter.export('exports/merged_' + model_type + '_dataframe.csv', used_data, selected_header)

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
        header_file_name = file_name + '_data_header.pkl'
        data_file_name = file_name + '_data.pkl'
        L.info('Converting data to single dataframe...')
        if not force_to_not_use_cache and self.cacher.file_available(header_file_name) and self.cacher.file_available(data_file_name):
            header = self.cacher.read_cache(header_file_name)
            data = self.cacher.read_cache(data_file_name)
        else:
            questionnaires = QuestionnaireFactory.construct_questionnaires(self.spss_reader)
            data, header = (self.single_output_frame_creator.create_single_frame(questionnaires, participants))
            self.cacher.write_cache(header, header_file_name)
            self.cacher.write_cache(data, data_file_name)

        return (header, data)
