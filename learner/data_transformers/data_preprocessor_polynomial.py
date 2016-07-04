from sklearn.preprocessing import PolynomialFeatures
from .data_transformer import DataTransformer
import numpy as np


class DataPreprocessorPolynomial(DataTransformer):

    def process(self, x, header, degree=2):
        poly = PolynomialFeatures(degree=degree)
        return poly.fit_transform(x)
