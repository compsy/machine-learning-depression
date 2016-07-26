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
        self.comm.Barrier()
        if self.rank == 0:
            L.info('\tStarting master')
            self.master()
        else:
            L.info('\t\tStarting slave')
            self.slave(X, y)

    def master(self):
        self.queue = Queue()

        temp = []
        for job in range(len(self.param_grid)):
            if (job % self.cpus_per_node == 0 and job != 0) or (job == (len(self.param_grid)-1)):
                self.queue.put(temp)
                temp = []
            current_job = self.merge_dicts([self.param_grid[job]])
            # current_job = self.param_grid[job]
            temp.append(current_job)

        # Add an extra job for each node to stop at the end
        for node in range(self.size - 1):
            self.queue.put(StopIteration)

        qsize = self.queue.qsize()
        status = MPI.Status()
        while not self.queue.empty():
            obj = self.queue.get()
            recv = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
            self.comm.send(obj=obj, dest=status.Get_source())
            L.info("\t-------------------")
            L.info("\tMaster: Queue size: %d/%d (last job by node %d, %d number of configurations, %d nodes)" % (self.queue.qsize(), qsize, recv,len(self.param_grid), self.size))
            # percent = ((position + 1) * 100) // (n_tasks + n_workers)
            # sys.stdout.write('\rProgress: [%-50s] %3i%% ' % ('=' * (percent // 2), percent))
            # sys.stdout.flush()
        L.info('\tQueue is empty, continueing')
        models = []
        models = self.comm.gather(models, root=0)
        best_model = 123
        best_score = float('-inf')
        
        L.info('\tWe received %d models' % len(models))
        none_models = 0
        empty_models = 0
        good_models = 0
        for model in models:
            if model is None:
                none_models += 1
                continue
            if len(model) == 0:
                empty_models += 1
                continue
            for sub_model in model:
                good_models += 1
                L.info(sub_model[0])
                L.info(sub_model[1])
                if sub_model[0] > best_score:
                    best_score = sub_model[0]
                    best_model = sub_model[1]
        L.info('\tThese models had %d nones, %d emptys, and %d goods' % (none_models, empty_models, good_models))
        return best_model

    def slave(self, X, y):
        models = []
        # Ask for work until we receive StopIteration
        L.info('\t\tSlave: Waiting for data..')
        for task in iter(lambda: self.comm.sendrecv(self.rank, 0), StopIteration):
            L.info('\t\tSlave: Picking up a task on node %d, task size: %d' % (self.rank, len(task)))
            model = GridSearchCV(estimator=self.skmodel, param_grid=task, n_jobs=-1, verbose=1, cv=self.cv)
            model = model.fit(X=X, y=y)

            # only add the best model
            L.info(model)
            model = (model.best_score_, model.best_estimator_)
            L.info(model)

            L.info('\t\t\t!!!!!!!!!! Appending model with score %d' % model[0])
            models.append(model)

        # Collective report to parent
        L.info('\t\tFinished calculating, calculated %d models' % len(models))
        self.comm.gather(sendobj=models, root=0)
        L.info('\t\tByebye')
        exit(0)
