import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from .data_transformer import DataTransformer

class DataResampler(DataTransformer):
    """
    Resamples under represented classes
    """

    @staticmethod
    def process(x_data, y_data):
        """Process
        This function will resample the data so that new observations with
        resampled data are created. This can help with highly unbalanced
        datasets, as is the case in the present work. See
        https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis
        for more details.
        :returns: the resampled X and Y
        """

        x_columns = x_data.columns
        y_columns = y_data.columns
        sm = SMOTEENN()
        x_data_resampled, y_data_resampled = sm.fit_sample(x_data, y_data)

        x_data_resampled = pd.DataFrame(x_data_resampled)
        x_data_resampled.columns = x_columns

        y_data_resampled = pd.DataFrame(y_data_resampled)
        y_data_resampled.columns = y_columns

        return (x_data_resampled, y_data_resampled)

