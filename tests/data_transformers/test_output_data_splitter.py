import numpy as np
from learner.data_transformers.output_data_splitter import OutputDataSplitter
import pytest


class TestOutputDataSplitter:
    @pytest.fixture()
    def subject(self):
        subject = OutputDataSplitter()
        return subject

    def test_split(self, subject):
        data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7],
                         [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11]])
        x_names = np.array(['input', 'mediator'])
        y_names = np.array(['output'])
        header = np.append(x_names, y_names)

        result = subject.split(data, header, x_names)
        expected = data[:, [0,1]]
        assert np.array_equal(result, expected)

        result = subject.split(data, header, y_names)
        expected = data[:, [2]]
        assert np.array_equal(result, expected)

        result = subject.split(data, header, header)
        expected = data
        assert np.array_equal(result, expected)
