from queue import Queue
import numpy as np
from data_output.std_logger import L


class SyncModelRunner:

    def __init__(self, models):
        self.models = models

    def fabricate_models(self, x, y, x_names, y_names, verbosity):
        L.info('Fabricating models')
        created_models = []
        for model in self.models:
            created_models.append(model(np.copy(x), np.copy(y), x_names, y_names, verbosity))
        return created_models

    def run_calculations(self, fabricated_models):
        for model in fabricated_models:
            L.info('Training from syncmodelrunner')
            model.train()

        return (True, fabricated_models)
