from queue import Queue
from threading import Thread
import numpy as np
from data_output.std_logger import L
from mpi4py import MPI

class DistributedModelRunner:

    def __init__(self, models):
        self.comm = MPI.COMM_WORLD
        self.models = models

    def fabricate_models(self, x, y, x_names, y_names, verbosity):
        created_models = []
        for model in self.models:
            created_models.append(model(np.copy(x), np.copy(y), x_names, y_names, verbosity))
        return created_models

    def run_calculations(self, fabricated_models):
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        data = []
        for i in range(len(fabricated_models)):
            if i == len(data):
                data.append([])
            data[i].append(fabricated_models[i])

        self.comm.scatter(data, root = 0)

        for model in data[self.rank]:
            L.info('Training from MPI model runner on node %d' % self.rank)
            model.train()
        self.comm.Barrier()
