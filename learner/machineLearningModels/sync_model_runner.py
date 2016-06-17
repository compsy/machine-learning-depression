from queue import Queue
import numpy as np


class SyncModelRunner:

    def __init__(self, models):
        self.models = models

    def runCalculations(self, x, y, x_names, y_names):
        self.result = Queue()
        for model in self.models:
            model = model(np.copy(x), np.copy(y), x_names, y_names).train()
            self.result.put((model, model.predict()))

        return self.result
