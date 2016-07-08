from queue import Queue
import numpy as np


class SyncModelRunner:

    def __init__(self, models):
        self.models = models

    def fabricate_models(self, x, y, x_names, y_names, verbosity):
        created_models = []
        for model in self.models:
            created_models.append(model(np.copy(x), np.copy(y), x_names, y_names, verbosity))
        return created_models

    def run_calculations(self, fabricated_models):
        self.result = Queue()
        for model in fabricated_models:
            print('\tTraining -> ' + model.given_name)
            self.result.put((model, model.train()))

        return self.result
