import numpy as np
from learner.data_transformers import output_data_cleaner


class TestOutputDataCleaner:

    def test_find_incomplete_rows(self):
        subject = output_data_cleaner.OutputDataCleaner()

        # With missings
        data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, np.nan], [7, 8],
                         [8, 9], [9, 10]])

        result = subject.find_incomplete_rows(data)
        expected = [5]
        assert np.array_equal(result, expected)

        # Without missings
        data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                         [8, 9], [9, 10]])
        result = subject.find_incomplete_rows(data)
        expected = []
        assert np.array_equal(result, expected)

    def test_clean(self):
        subject = output_data_cleaner.OutputDataCleaner()

        # With missings
        data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, np.nan], [7, 8],
                         [8, 9], [9, 10]])

        result = subject.clean(data, [5])
        expected = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10]])

        assert len(result) == (len(data) - 1)
        assert np.array_equal(result, expected)