from queue import Queue
import numpy as np
from data_output.std_logger import L


class SyncModelRunner:

    def __init__(self, models, hpc=False):
        self.models = models
        self.hpc = hpc

    def fabricate_models(self, x, y, x_names, y_names, verbosity):
        L.info('Fabricating models')
        created_models = []
        for model in self.models:
            created_models.append(model(np.copy(x), np.copy(y), x_names, y_names, verbosity, hpc=self.hpc))
        return created_models

    def run_calculations(self, fabricated_models):
        result = True
        for model in fabricated_models:
            L.info('Training from syncmodelrunner')
            result = model.train()

        # Convert the outcome to a boolean
        result = result != False
        return (result, fabricated_models)
