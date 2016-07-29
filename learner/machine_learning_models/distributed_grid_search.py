from sklearn.grid_search import GridSearchCV, ParameterGrid
from queue import Queue
from data_output.std_logger import L
from mpi4py import MPI
from machine_learning_models.grid_search_mine import GridSearchMine
import random
import math



class DistributedGridSearch:

    def __init__(self, estimator, param_grid, cv):
        # Number of nodes
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.workers = self.size -1
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
        if self.rank == 0:
            L.info('\tStarting master')
            return self.master()
        else:
            self.slave(X, y)
            return False

    def create_job_queue(self, shuffle):
        queue = Queue()
        shuffled_range = list(range(len(self.param_grid)))
        if shuffle: random.shuffle(shuffled_range)

        work_division = self.cpus_per_node
        if(self.cpus_per_node * self.workers > len(self.param_grid)):
            work_division = math.ceil(len(self.param_grid) / self.workers)

        temp = []
        for job in shuffled_range:
            temp.append(job)
            # current_job = self.merge_dicts([self.param_grid[job]])
            # current_job = self.param_grid[job]
            # temp.append(current_job)
            if (len(temp) == work_division):
                queue.put(temp)
                temp = []

        if len(temp) is not 0: queue.put(temp)

        # Add an extra job for each node to stop at the end
        for node in range(self.workers):
            queue.put(StopIteration)

        return queue

    def master(self):

        # Get the queue of jobs to create
        queue = self.create_job_queue(shuffle=True)
        send_queue = Queue()
        qsize = queue.qsize()

        status = MPI.Status()
        # running_procs = set()
        wt = MPI.Wtime()
        total_wait_time = []

        while not queue.empty() or not send_queue.empty():
            L.info('\tWaiting for worker...')
            recv = self.comm.recv(source=MPI.ANY_SOURCE, status=status)

            # If it receives a next, reply with a job
            if recv == 'next':
                send_queue.put(status.Get_source())
                obj = queue.get()
                self.comm.send(obj=obj, dest=status.Get_source())
                L.info("\t-------------------")
                L.info("\tMaster: Sending to node %d: %s" % (status.Get_source(), obj))
                L.info("\tMaster: Queue size: %d/%d (last job by node %d, %d number of configurations, %d nodes)" % (queue.qsize(), qsize, status.Get_source(),len(self.param_grid), self.workers))
                L.info("\t-------------------")
            else:
                L.info("\tMaxter: Done: %d in %0.2f seconds" % (send_queue.get(), recv))
                total_wait_time.append(recv)

        wt = MPI.Wtime() - wt
        L.info('\tQueue is empty, it contained %d items which took %0.2f seconds (%0.2f minutes) continueing (%s time spent good)' % (qsize, wt, (wt/60), total_wait_time))

        models = []
        models = self.comm.gather(models, root=0)
        best_model = None
        best_score = float('-inf')

        L.info('\tWe received %d models' % len(models))
        good_models = 0
        for model in models:
            if len(model) == 0: continue
            for sub_model in model:
                good_models += 1
                if sub_model[0] > best_score:
                    best_score = sub_model[0]
                    best_model = sub_model[1]

        L.info('\tThese models had %d good models' % (good_models))
        L.info('\tThe score of the best model was %0.3f' % best_score)
        return best_model

    def slave(self, X, y):
        models = []
        my_wait_time=0
        my_run_time=0
        prev = MPI.Wtime()
        prev_run=0
        # Ask for work until we receive StopIteration
        print('\t\tSlave: Waiting for data..')
        for task in iter(lambda: self.comm.sendrecv('next', 0), StopIteration):
        #for task in iter(lambda: self.comm.recv(source=0), StopIteration):
            my_wait_time += (MPI.Wtime() - prev)
            prev_run = MPI.Wtime()
            grid = [self.param_grid[y] for y in task]
            # L.info('\t\tSlave: Picking up a task on node %d, task size: %d' % (self.rank, len(task)))
            print('\t\tSlave %d: starting calculation' % self.rank)
            model = GridSearchMine(estimator=self.skmodel, param_grid=grid, n_jobs=-1, verbose=0, cv=self.cv)
            print('\t\tSlave %d: finished calculation' % self.rank)
            model = model.fit(X=X, y=y)
            model = (model.best_score_, model.best_estimator_)
            models.append(model)
            my_run_time += MPI.Wtime() - prev_run
            prev = MPI.Wtime()
         #   self.comm.send(obj='next', dest=0)

        my_wait_time += (MPI.Wtime() - prev)
        self.comm.send(obj=my_run_time, dest=0)
        # Collective report to parent
        # L.info('\t\tFinished calculating (node %d), calculated %d models' % (self.rank, len(models)), force=True)
        self.comm.gather(sendobj=models, root=0)
        # L.info('\t\tByebye')
        #exit(0)
