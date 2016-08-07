from sklearn.grid_search import GridSearchCV, ParameterGrid, RandomizedSearchCV
import os.path
from queue import Queue
from data_output.std_logger import L
from mpi4py import MPI
from machine_learning_models.grid_search_mine import GridSearchMine
import random
import math
import numpy as np



class DistributedRandomGridSearch:

    def __init__(self, ml_model, estimator, param_grid, cv, n_iter=10000):
        # Number of nodes
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.root = self.rank == 0
        self.cpus_per_node = 23
        self.skmodel = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.ml_model = ml_model
        self.iterations = n_iter

    def fit(self, X, y):
        my_X = np.copy(X)
        my_y = np.copy(y)

        if (self.root):
            iterations = [round(self.iterations / self.size)] * self.size
        else:
            iterations = np.empty(self.size)

        self.comm.Barrier()

        L.info('Running %d iterations on %d nodes.' % (len(iterations), self.size))
        iterations = self.comm.scatter(iterations, root=0)

        # Actual calculation
        my_data = []
        my_iterations = iterations[0]
        model = RandomizedSearchCV(estimator=self.skmodel, param_distributions=self.param_grid,
                                   n_jobs=-1, verbose=0, cv=self.cv, n_iter=my_iterations)
        model = model.fit(X=my_X, y=my_y)
        L.info('Training from MPI model runner on node %d with %d iterations' % (self.rank, my_iterations))
        my_data.append((model.best_score_, model.best_estimator_))

        iterations = my_data

        self.comm.Barrier()

        L.info('!!Trained all models!!')

        iterations = self.comm.gather(iterations, root=0)

        if not self.root: return False

        best_model = None
        best_score = float('-inf')
        L.info('\tWe received %d models' % len(iterations))
        good_models = 0
        for model in iterations:
            if len(model) == 0: continue
            for sub_model in model:
                good_models += 1
                if sub_model[0] > best_score:
                    best_score = sub_model[0]
                    best_model = sub_model[1]

        L.info('\tThese models had %d good models' % (good_models))
        L.info('\tThe score of the best model was %0.3f' % best_score)
        return best_model