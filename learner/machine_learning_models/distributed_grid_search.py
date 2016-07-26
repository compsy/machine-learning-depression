from sklearn.grid_search import GridSearchCV, ParameterGrid
from queue import Queue
from data_output.std_logger import L
from mpi4py import MPI


class DistributedGridSearch:

    def __init__(self, estimator, param_grid, cv):
        # Number of nodes
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.cpus_per_node = 12
        self.skmodel = estimator
        self.param_grid = ParameterGrid(param_grid)
        self.cv = cv

    def merge_dicts(self, dicts):
        result = {}
        for key in dicts[0].keys():
            result[key] = [d[key] for d in dicts]
        return result

    def fit(self, X, y):
        # Sync all nodes

        # self.comm.send(obj=1, dest=0)
        # if self.rank == 0:
            # a = 0
            # running = True
            # while (self.comm.recv() and running):
                # a += 1
                # print('%d of %d' % (a, self.size))
                # if a == self.size: running = False
        
        L.info('Approaching barrier')
        if self.rank == 0:
            L.info('Starting master')
            self.master()
        else:
            L.info('Starting slave')
            self.slave(X, y)

    def master(self):
        self.queue = Queue()

        for job in range(len(self.param_grid)):
            temp = []
            for jobs_per_node in range(self.cpus_per_node):
                temp.append(self.param_grid[job])
            temp = self.merge_dicts(temp)
            self.queue.put(temp)

        # Add an extra job for each node to stop at the end
        for node in range(self.size):
            self.queue.put(StopIteration)

        status = MPI.Status()
        while not self.queue.empty():
            obj = self.queue.get()
            recv = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
            print(recv)
            self.comm.send(obj=obj, dest=status.Get_source())
            L.info(self.queue.qsize())
            # percent = ((position + 1) * 100) // (n_tasks + n_workers)
            # sys.stdout.write('\rProgress: [%-50s] %3i%% ' % ('=' * (percent // 2), percent))
            # sys.stdout.flush()

        models = self.comm.gather(root=MPI.ROOT)
        best_model = None
        best_score = float('-inf')
        for model in models:
            if model[0] > best_score:
                best_score = model[0]
                best_model = model[1]

        return best_model

    def slave(self, X, y):
        models = []
        # Ask for work until we receive StopIteration
        L.info('Waiting for data..')
        for task in iter(lambda: self.comm.sendrecv(9, 0), StopIteration):
            L.info('Picking up a task on node %d' % self.rank)
            model = GridSearchCV(estimator=self.skmodel, param_grid=self.param_grid, n_jobs=-1, verbose=1, cv=self.cv)
            model = model.fit(X=X, y=y)

            # only add the best model
            model = (model.best_score_, model.best_estimator_)
            models.append(model)

        # Collective report to parent
        self.comm.gather(sendobj=models, root=0)
        exit(0)
