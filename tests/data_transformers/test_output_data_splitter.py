import numpy as np
from learner.data_transformers import output_data_splitter


class TestOutputDataSplitter:

    def test_split(self):
        data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]])
        x_names = np.array(['input'])
        y_names = np.array(['output'])
        header = np.append(x_names, y_names)
        subject = output_data_splitter.OutputDataSplitter()
        result = subject.split(data, header, x_names)
        expected = data[:, [0]]
        assert np.array_equal(result, expected)

        result = subject.split(data, header, y_names)
        expected = data[:, [1]]
        assert np.array_equal(result, expected)

        result = subject.split(data, header, header)
        expected = data
        assert np.array_equal(result, expected)
