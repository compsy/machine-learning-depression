from queue import Queue
from threading import Thread
import numpy as np
from data_output.std_logger import L
from mpi4py import MPI

class DistributedModelRunner:

    def __init__(self, models):
        L.info('Running distributed model runner')
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        L.info(self.comm)
        L.info('This is node %d' % self.rank)
        self.models = models

    def fabricate_models(self, x, y, x_names, y_names, verbosity):
        L.info('Fabbing models')
        created_models = []
        for model in self.models:
            L.info('Model')
            created_models.append(model(np.copy(x), np.copy(y), x_names, y_names, verbosity))
        return created_models

    def run_calculations(self, fabricated_models):
        data = []
        for i in range(len(fabricated_models)):
            if i == len(data):
                data.append([])
            data[i].append(fabricated_models[i])
        if(self.rank == 0): L.info('Running %d models on %d nodes' % (len(data), self.size))

        self.comm.scatter(data, root = 0)

        for model in data[self.rank]:
            L.info('Training from MPI model runner on node %d' % self.rank)
            model.train()
        self.comm.Barrier()
