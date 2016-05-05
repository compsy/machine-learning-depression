from queue import Queue
from threading import Thread
import numpy as np


class AsyncModelRunner:
    def __init__(self, models, workers = 8):
        self.models = models
        self.workers = workers

    def runCalculations(self, data, headers, x, y):
        self.result = Queue()
        self.queue = Queue()
        for model in self.models:
            self.queue.put(model)

        workers = min([self.workers, self.queue.qsize()])
        print('Starting calculation of models with %s workers.' % workers )
        for i in range(workers):
            t = Thread(target=self.work, args=(np.copy(data), np.copy(headers), np.copy(x), np.copy(y)))
            t.daemon = True
            t.start()

        self.queue.join()
        return self.result

    def work(self, data, headers, x, y):
        while True:
            model = self.queue.get()
            model = model(data,headers,x,y)
            self.result.put((model, model.train()))
            self.queue.task_done()


