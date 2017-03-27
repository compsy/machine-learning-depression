import numpy as np
from learner.data_output.std_logger import L
from learner.machine_learning_models.models.bagging_model import BaggingClassificationModel, BaggingModel


class ModelRunner:

    def __init__(self, models, hpc):
        self.models = []
        self.model_options = []
        self.hpc = hpc
        for model in models:
            self.models.append(model['model'])
            self.model_options.append(model['options'])

    def run_calculations(self, fabricated_models):
        raise NotImplementedError('This function needs to be implemented in the subclass')

    def fabricate_models(self, x, y, x_names, y_names, verbosity):
        """
        Fabricates (that is, creates instances) of the models in self.models. Furthermore, it will enable gridsearch
        and bagging for those models if specified.

        :param x: the training x values
        :param y: the training y values
        :param x_names: the names of the x data
        :param y_names: the names (variable names) of the y data
        :param verbosity: whether it has to be verbose or not
        :return: a list of fabricated / instances of ML models
        """
        L.info('Fabricating models')
        created_models = []

        for i, model in enumerate(self.models):
            # Get the options for the current model
            has_grid_search = True if ('grid-search' in self.model_options[i]) else False

            current_model = model(
                np.copy(x),
                np.copy(y),
                x_names,
                y_names,
                verbosity=verbosity,
                grid_search=has_grid_search)

            if ('bagging' in self.model_options[i]):
                current_model.skmodel = self.use_bagging(current_model, verbosity)

            created_models.append(current_model)

        L.info('Done fabricating')
        return created_models

    def use_bagging(self, model, verbosity):
        current_skmodel = None
        if (model.model_type == 'regression'):
            current_skmodel = BaggingModel
        elif (model.model_type == 'classification'):
            current_skmodel = BaggingClassificationModel

        return current_skmodel.use_bagging(verbosity=verbosity, skmodel=model.skmodel)
