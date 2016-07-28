from queue import Queue
from threading import Thread
import numpy as np
from data_output.std_logger import L


class AsyncModelRunner:

    def __init__(self, models, workers=8):
        self.models = models
        self.workers = workers

    def runCalculations(self, x, y, x_names, y_names):
        self.result = Queue()
        self.queue = Queue()
        for model in self.models:
            self.queue.put(model)

        workers = min([self.workers, self.queue.qsize()])
        L.info('Starting calculation of models with %s workers.' % workers)
        for i in range(workers):
            t = Thread(target=self.work, args=(np.copy(x), np.copy(y), x_names, y_names))
            t.daemon = True
            t.start()

        self.queue.join()
        return self.result

    def work(self, x, y, x_names, y_names):
        while True:
            model = self.queue.get()
            model = model(x, y, x_names, y_names).train()
            self.result.put((model, model.predict()))
            self.queue.task_done()
