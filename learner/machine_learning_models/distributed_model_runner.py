import numpy as np
from data_output.std_logger import L
from mpi4py import MPI

class DistributedModelRunner:

    def __init__(self, models):
        L.info('Running distributed model runner')
        self.models = models

    def fabricate_models(self, x, y, x_names, y_names, verbosity):
        # self.comm = MPI.COMM_WORLD
        # self.size = self.comm.Get_size()
        # self.rank = self.comm.Get_rank()
        # L.info('This is node %d/%d' % (self.rank, self.size))
        # self.models = [1, 2]
        # L.info('Fabbing models')
        #
        # jobs_per_node = 3
        #
        # if (self.rank == 0):
        #     data = self.models
        #     # data = []
        #     # for i in range(len(self.models)):
        #     #     if i == len(data):
        #     #         data.append([])
        #     #     data[i].append(self.models[i])
        #     # data = self.models
        #     # dat = ', '.join(data)
        # else:
        #     data = np.empty(len(self.models))
        #
        # my_data = np.empty(jobs_per_node)
        # self.comm.Barrier()
        # if (self.rank == 0): L.info('Running %d models on %d nodes (%d jobs per node)' % (len(data), self.size, len(my_data)))
        #
        # data = self.comm.scatter(data, root=0)
        #
        # model = data
        # #for model in data:
        # L.info('Training from MPI model runner on node %d' % self.rank)
        # model = model(np.copy(x), np.copy(y), x_names, y_names, verbosity)
        # model.train()
        #
        # self.comm.Barrier()
        #
        # if self.rank == 0: L.info('!!Trained all models!!')
        #
        # return self.comm.Gather(data, root=0)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0: print("-" * 78)
        if rank == 0: print(" Running on %d cores" % comm.size)
        if rank == 0: print("-" * 78)

        my_N = 1
        N = my_N * comm.size

        if comm.rank == 0:
            A = np.arange(N, dtype=np.float64)
        else:
            A = np.empty(N, dtype=np.float64)

        my_A = np.empty(my_N, dtype=np.float64)

        # Scatter data into my_A arrays
        comm.Scatter([A, MPI.DOUBLE], [my_A, MPI.DOUBLE])

        if rank == 0: print("After Scatter:")
        for r in range(comm.size):
            if comm.rank == r:
                print("[%d] %s" % (comm.rank, my_A))
            comm.Barrier()

        # Everybody is multiplying by 2
        my_A *= 2

        # Allgather data into A again
        comm.Allgather([my_A, MPI.DOUBLE], [A, MPI.DOUBLE])

        if rank == 0: print("After Allgather:")
        for r in range(comm.size):
            if comm.rank == r:
                print("[%d] %s" % (comm.rank, A))
            comm.Barrier()


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
