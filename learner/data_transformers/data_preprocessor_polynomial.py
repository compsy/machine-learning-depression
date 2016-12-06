from sklearn.preprocessing import PolynomialFeatures
from .data_transformer import DataTransformer
import numpy as np


class DataPreprocessorPolynomial(DataTransformer):

    def process(self, x, header, degree=2):
        """Process
        it will add polynomials to the provided dataframe
        if the degree provided is 2, it will add a const row of 1, a 1st degree polynomial
        (the original data), and a 2nd degree polynomial.  Furthermore, it will add interaction terms
        :returns: a new dataframe with polynomial features

        """
        poly = PolynomialFeatures(degree=degree)
        return poly.fit_transform(x)
