import numpy as np
from learner.data_output.std_logger import L
from learner.machine_learning_models.models.bagging_model import BaggingClassificationModel, BaggingModel


class ModelRunner:

    def __init__(self, models):
        self.models = []
        self.model_options = []
        for model in models:
            self.models.append(model['model'])
            self.model_options.append(model['options'])

    def fabricate_models(self, x, y, x_names, y_names, verbosity):
        L.info('Fabricating models')
        created_models = []
        for i, model in enumerate(self.models):
            # Get the options for the current model
            current_model = model(np.copy(x), np.copy(y), x_names, y_names, verbosity, hpc=self.hpc)
            if ('bagging' in self.model_options[i]):
                bagged_sk_model = self.use_bagging(current_model, verbosity)
                current_model.skmodel = bagged_sk_model

            created_models.append(current_model)
        return created_models

    def use_bagging(self, model, verbosity):
        current_skmodel = None
        if (model.model_type == 'regression'):
            current_skmodel = BaggingModel
        elif (model.model_type == 'classification'):
            current_skmodel = BaggingClassificationModel

        return current_skmodel.use_bagging(verbosity=verbosity, skmodel=model.skmodel)
