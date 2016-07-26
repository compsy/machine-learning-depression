from sklearn.grid_search import GridSearchCV, ParameterGrid
from queue import Queue

class DistributedGridSearch:
    def __init__(self, estimator, param_grid, cv):
        # Number of nodes
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.cpus_per_node = 12
        self.skmodel = estimator
        self.param_grid = param_grid
        self.cv = cv

    def merge_dicts(self, dicts):
        result = {}
        for key in dicts[0].keys():
            result[key] = [d[key] for d in dicts]
        return result

    def fit(self, X, y):
        # Sync all nodes
        self.comm.Barrier()
        if self.rank == 0:
            self.master()
        else:
            self.slave(X, y)

    def master(self):
        self.queue = Queue()

        for job in range(len(param_grid)):
            temp = []
            for jobs_per_node in range(self.cpus_per_node):
                temp.append(self.param_grid[job])
            temp = self.merge_dicts(temp)
            self.queue.put(temp)

        # Add an extra job for each node to stop at the end
        for node in range(self.size):
            self.queue.put(StopIteration)

        status = MPI.Status()
        while(not self.queue.empty):
            obj = self.queue.get()
            self.comm.recv(source=MPI.ANY_SOURCE, status=status)
            self.send(obj=obj, dest=status.Get_source())
            percent = ((position + 1) * 100) // (n_tasks + n_workers)
            sys.stdout.write('\rProgress: [%-50s] %3i%% ' %
                    ('=' * (percent // 2), percent))
            sys.stdout.flush()

        models = comm.gather(root=MPI.ROOT)
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
        for task in iter(lambda: comm.sendrecv(dest=0), StopIteration):
            model = GridSearchCV(estimator=self.skmodel,
                    param_grid=self.param_grid, n_jobs=-1, verbose=1,
                    cv=self.cv)
            model = model.fit(X=X, y=y)

            # only add the best model
            model = (model.best_score_, model.best_estimator_)
            models.append(model)

        # Collective report to parent
        comm.gather(sendobj=models, root=0)

a = {'kernel': ['a','b','c','d','e'], 'test': [1,2,3,4,5], 'test2':[6,7,8], 'test3':[9,0,11]}
param_grid = ParameterGrid(a)
DistributedGridSearch(param_grid)
