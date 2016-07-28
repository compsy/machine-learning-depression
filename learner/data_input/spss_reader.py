import pandas as pd
# import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
import warnings


class SpssReader:

    def read_file(self, filename):
        with warnings.catch_warnings():
            rpackages.importr('foreign')
            read_spss = robjects.r['read.spss']
            data = read_spss("../data/" + filename, to_data_frame=True, use_value_labels=False, reencode='utf-8')
            data = robjects.DataFrame(data)
            data = pandas2ri.ri2py(data)
            return data
