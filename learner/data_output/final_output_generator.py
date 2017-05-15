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


class OutputGenerator():

    def __init__(self):
        # # Create objects to perform image plotting
        self.actual_vs_prediction_plotter = ActualVsPredictionPlotter()
        self.learning_curve_plotter = LearningCurvePlotter()
        self.validation_curve_plotter = ValidationCurvePlotter()
        self.roc_curve_plotter = RocCurvePlotter()
        self.confusion_matrix_plotter = ConfusionMatrixPlotter()
        self.data_density_plotter = DataDensityPlotter()
        self.cacher = ObjectCacher(MachineLearningModel.cache_directory())

    def create_output(self, classification_fabricated_models, y_data, used_data, selected_header, output_type,  model_type='classification'):
        models = []
        for model in classification_fabricated_models:
            cache_name = model.model_cache_name
            files = self.cacher.files_in_dir()

            # Only use the hyperparameters of the for the present model
            files = list(filter(lambda x: cache_name in x, files))

            best_score = 0
            skmodel = None
            for filename in files:
                cached_params = self.cacher.read_cache(filename)
                if cached_params['skmodel'] is None:
                    continue
                if cached_params['score'] > best_score:
                    best_score = cached_params['score']
                    skmodel = cached_params['skmodel']

            if skmodel is not None:
                model.inject_trained_model(skmodel=skmodel)
                models.append(model)
            else:
                L.info('File %s has none' % cache_name)

        if len(models) == 0:
            raise ValueError('There are no methods for printing the evaluation of. Something went wrong...')

        if model_type == 'classification':
            outcome = models[0].y.mean()
            L.info('In the %s set, %d participants (%0.2f percent) is true' % (output_type, (len(models[0].y)), outcome))

            # Generate roc curve plots
            self.roc_curve_plotter.plot(models, output_type=output_type)

        # Export all used data to a CSV file
        CsvExporter.export('exports/merged_' + model_type + '_' + output_type +'_dataframe.csv', used_data, selected_header)

        # for model in models:
            # learning_curve_plotter.plot(model)
            # validation_curve_plotter.plot(model, variable_to_validate='n_estimators')

        for model in models:
            model.print_evaluation()
            y_test_pred = model.skmodel.predict(model.x)

            if model_type == 'classification':
                self.confusion_matrix_plotter.plot(model, model.y, y_pred, output_type=output_type)
            else:
                self.actual_vs_prediction_plotter.plot_both(model, model.y_test, y_test_pred, model.y_train,
                                                            y_train_pred)