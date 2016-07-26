import numpy as np
from data_output.std_logger import L
from mpi4py import MPI


class DistributedModelRunner:

    def __init__(self, models):
        L.info('Running distributed model runner')
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        L.info('This is node %d/%d' % (self.rank, self.size))
        self.models = models
        self.is_root = True if self.rank == 0 else False

    def fabricate_models(self, x, y, x_names, y_names, verbosity):
        L.info('Fabricating models')
        if self.is_root:
            self.data = []
            for i in range(len(self.models)):
                if i == len(self.data):
                    self.data.append([])
                self.data[i].append(self.models[i])
        else:
            self.data = np.empty(len(self.models))

        return self.data

    def run_calculations(self, fabricated_models):
        # Be sure eveyone gets here, before running the calculations
        self.comm.Barrier()
        if self.is_root: L.info('Running %d models on %d nodes.' % (len(fabricated_models), self.size))

        # Distribute the data to all workers
        data = self.comm.scatter(fabricated_models, root=0)

        #model = data
        my_data = []
        for model in data:
            L.info('Training from MPI model runner on node %d' % self.rank)
            model = model(np.copy(x), np.copy(y), x_names, y_names, verbosity)
            model.train()
            my_data.append(model)

        data = my_data

        self.comm.Barrier()

        if self.is_root: L.info('Trained all %d models' % len(self.models))
        if self.is_root: L.info('Gathering data from all worker nodes')

        data = self.comm.gather(data, root=0)

        if not self.is_root: return (self.is_root, None)

        data = [val for sublist in data for val in sublist]
        return (self.is_root, data)
