from learner.caching.object_cacher import ObjectCacher
from learner.data_output.csv_exporter import CsvExporter
from learner.data_output.plotters.actual_vs_prediction_plotter import ActualVsPredictionPlotter
from learner.data_output.plotters.confusion_matrix_plotter import ConfusionMatrixPlotter
from learner.data_output.plotters.data_density_plotter import DataDensityPlotter
from learner.data_output.plotters.learning_curve_plotter import LearningCurvePlotter
from learner.data_output.plotters.roc_curve_plotter import RocCurvePlotter
from learner.data_output.plotters.validation_curve_plotter import ValidationCurvePlotter
from learner.data_output.std_logger import L
from learner.machine_learning_models.machine_learning_model import MachineLearningModel
from learner.caching.cacher import Cacher


class OutputGenerator():
    """
    Class that generates output files / log files for all models
    """

    def __init__(self):
        # # Create objects to perform image plotting
        self.actual_vs_prediction_plotter = ActualVsPredictionPlotter()
        self.learning_curve_plotter = LearningCurvePlotter()
        self.validation_curve_plotter = ValidationCurvePlotter()
        self.roc_curve_plotter = RocCurvePlotter()
        self.confusion_matrix_plotter = ConfusionMatrixPlotter()
        self.data_density_plotter = DataDensityPlotter()
        self.cacher = ObjectCacher(MachineLearningModel.cache_directory())

    def create_output(self, fabricate_models, x_data, y_data, output_type, model_type='classification'):
        """
        Create output files for the test / training set
        """
        algorithms = []
        for algorithm in fabricate_models:
            cache_name = algorithm.short_name
            files = self.cacher.files_in_dir()

            # Only use the hyperparameters of the for the present algorithm
            files = [file_name for file_name in files if cache_name in file_name]

            best_score = 0
            skmodel = None
            for filename in files:
                cached_params = self.cacher.read_cache(filename)
                if not Cacher.is_valid_cache(cached_params, ['skmodel']) or cached_params['skmodel'] is None:
                    continue
                if cached_params['score'] > best_score:
                    best_score = cached_params['score']
                    skmodel = cached_params['skmodel']

            if skmodel is not None:
                algorithm.inject_trained_model(skmodel=skmodel)
                algorithm.x = x_data
                assert all(algorithm.get_x == x_data)
                algorithm.y = y_data
                assert all(algorithm.get_y == y_data)
                algorithms.append(algorithm)
            else:
                L.info('File %s has no skmodel (not calculated)' % cache_name)

        if len(algorithms) == 0:
            raise ValueError('There are no methods for printing the evaluation of. Something went wrong...')

        if model_type == 'classification':
            outcome = y_data.mean()
            L.info('In the %s set, %d participants (%0.2f percent) is true' % (output_type, (len(y_data)), outcome))

            # Generate roc curve plots
            self.roc_curve_plotter.plot(algorithms, output_type=output_type)

        for algorithm in algorithms:
            algorithm.print_evaluation()
            y_test_pred = algorithm.skmodel.predict(x_data)

            # if model_type == 'classification':
            self.confusion_matrix_plotter.plot(
                algorithm, y_data, y_test_pred, output_type=output_type
            )
            # else:
                # self.actual_vs_prediction_plotter.plot_both(
                    # algorithm, algorithm.y_test, y_test_pred, algorithm.y_train, y_train_pred
                # )
