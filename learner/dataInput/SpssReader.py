import pandas as pd
# import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages

class SpssReader:
    def read_file(self, filename):
        rpackages.importr('foreign')
        read_spss = robjects.r['read.spss']
        data = read_spss("../data/"+filename, to_data_frame=True, use_value_labels=False)
        data = robjects.DataFrame(data)
        data = pandas2ri.ri2py(data)
        return data
