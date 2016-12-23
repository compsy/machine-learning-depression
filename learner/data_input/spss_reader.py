import pandas as pd
# import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
import warnings
from learner.data_output.std_logger import L


class SpssReader:

    def __init__(self):
        rpackages.importr('foreign')
        self.read_spss = robjects.r['read.spss']
        self.base_dir = 'data/'

    def read_file(self, filename):
        with warnings.catch_warnings():
            L.info('Reading %s' % filename)
            data = self.read_spss(
                self.base_dir + filename, to_data_frame=True, use_value_labels=False, reencode='utf-8')
            data = robjects.DataFrame(data)
            data = pandas2ri.ri2py(data)
            return data
