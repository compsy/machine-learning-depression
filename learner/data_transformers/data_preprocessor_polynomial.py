from sklearn.preprocessing import PolynomialFeatures
from .data_transformer import DataTransformer
import numpy as np
import pandas as pd

class DataPreprocessorPolynomial(DataTransformer):
    """
    Converts data to have polynomial interaction terms
    """

    @staticmethod
    def process(x_data, degree=2):
        """Process
        it will add polynomials to the provided dataframe
        if the degree provided is 2, it will add a const row of 1, a 1st degree polynomial
        (the original data), and a 2nd degree polynomial.  Furthermore, it will add interaction terms
        :returns: a new dataframe with polynomial features
        """

        poly = PolynomialFeatures(degree=degree)
        fit = poly.fit_transform(x_data)
        names = poly.get_feature_names(list(x_data))

        return pd.DataFrame(fit, columns=names)
