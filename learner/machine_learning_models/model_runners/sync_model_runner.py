from queue import Queue
import numpy as np
from learner.data_output.std_logger import L
from learner.machine_learning_models.model_runner import ModelRunner


class SyncModelRunner(ModelRunner):

    def __init__(self, models, hpc=False):
        super().__init__(models)
        self.hpc = hpc

    def run_calculations(self, fabricated_models):
        result = True
        for model in fabricated_models:
            L.info('Training from syncmodelrunner')
            result = model.train()

        # Convert the outcome to a boolean
        result = result != False
        return (result, fabricated_models)
