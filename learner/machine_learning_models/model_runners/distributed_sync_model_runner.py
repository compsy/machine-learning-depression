import numpy as np
from data_output.std_logger import L
from mpi4py import MPI

from machine_learning_models.model_runner import ModelRunner


class DistributedModelRunner(ModelRunner):

    def __init__(self, models):
        super().__init__(models)
        L.info('Running distributed model runner')
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        L.info('This is node %d/%d' % (self.rank, self.size))

    def fabricate_models(self, x, y, x_names, y_names, verbosity):
        L.info('Fabbing models')

        state = True if self.rank == 0 else False

        if (self.rank == 0):
            data = []
            for i in range(len(self.models)):
                if i == len(data):
                    data.append([])
                data[i].append(self.models[i])
        else:
            data = np.empty(len(self.models))

        self.comm.Barrier()
        if (self.rank == 0): L.info('Running %d models on %d nodes.' % (len(data), self.size))

        data = self.comm.scatter(data, root=0)

        #model = data
        my_data = []
        for model in data:
            L.info('Training from MPI model runner on node %d' % self.rank)
            model = model(np.copy(x), np.copy(y), x_names, y_names, verbosity)
            model.train()
            my_data.append(model)

        data = my_data

        self.comm.Barrier()

        if self.rank == 0: L.info('!!Trained all models!!')

        data = self.comm.gather(data, root=0)

        if not state: return (state, data)

        data = [val for sublist in data for val in sublist]
        L.info(data)
        L.info(len(data))

        return (state, data)

    def run_calculations(self, fabricated_models):

        if (self.rank == 0):
            data = []
            for i in range(len(fabricated_models)):
                if i == len(data):
                    data.append([])
                data[i].append(fabricated_models[i])
        else:
            data = None

        self.comm.Barrier()

        L.info(data)
        if (self.rank == 0): L.info('Running %d models on %d nodes' % (len(data), self.size))

        data = self.comm.scatter(data, root=0)

        L.info('Yes here! from %d' % self.rank)

        for model in data[self.rank]:
            L.info('Training from MPI model runner on node %d' % self.rank)
            model.train()
        self.comm.Barrier()

        if self.rank == 0: L.info('!!Trained all models!!')

        return self.comm.gather(data, root=0)
