from .data_transformer import DataTransformer
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np

class ScalingTransformer(DataTransformer):
    @staticmethod
    def perform_scaling(data, scale_binary=False):

        # First we fit the encoder to determine the number of options in each variable

        usable_columns = data.columns
        if not scale_binary:
            usable_columns = []
            for colname in data.columns:
                is_binary = np.array_equal(np.sort(pd.Series.unique(data[colname])), [0,1])
                if not is_binary:
                    usable_columns.append(colname)

        np_data = scale(data[usable_columns])
        scaled_data = pd.DataFrame(np_data, columns=usable_columns)
        not_scaled_data = data[data.columns.difference(usable_columns)]

        # Merge the two datatables
        return pd.concat([scaled_data, not_scaled_data], axis=1)
