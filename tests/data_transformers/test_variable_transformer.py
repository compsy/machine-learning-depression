import numpy as np
import pytest

from learner.data_transformers.variable_transformer import VariableTransformer


class TestVariableTransformer:

    @pytest.fixture()
    def subject(self, mock_reader):
        header = np.array(['a', 'b', 'c'])
        subject = VariableTransformer(header)
        return subject

    def test_log_transform(self, subject):
        # It should log transform the data
        data = np.array([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.], [5., 6., 7.], [6., 7., 8.],
                         [7., 8., 9.], [8., 9., 10.], [9., 10., 11.]])
        orig_data = data.copy()
        subject.log_transform(data, 'a')
        for val in range(np.shape(data)[0]):
            assert data[val, 0] == np.log(orig_data[val, 0])
            assert data[val, 1] == orig_data[val, 1]
            assert data[val, 2] == orig_data[val, 2]

        subject.log_transform(data, 'b')
        for val in range(np.shape(data)[0]):
            assert data[val, 0] == np.log(orig_data[val, 0])
            assert data[val, 1] == np.log(orig_data[val, 1])
            assert data[val, 2] == orig_data[val, 2]

        # It should raise if a non existing field is provided
        with pytest.raises(ValueError):
            subject.log_transform(data, 'test')

        # It should add a constant if the data contains a zero
        data = np.array([[0., 0., 0.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.], [5., 6., 7.], [6., 7., 8.],
                         [7., 8., 9.], [8., 9., 10.], [9., 10., 11.]])
        orig_data = data.copy()
        subject.log_transform(data, 'a')
        for val in range(np.shape(data)[0]):
            assert data[val, 0] == np.log(orig_data[val, 0] + 1)
            assert data[val, 1] == orig_data[val, 1]
            assert data[val, 2] == orig_data[val, 2]

        # It should never be able to contain a zero
        data = np.array([[0., 0., 0.], [-2., -3., -4.], [3., 4., 5.], [4., 5., 6.], [5., 6., 7.], [6., 7., 8.],
                         [7., 8., 9.], [8., 9., 10.], [9., 10., 11.]])
        orig_data = data.copy()
        subject.log_transform(data, 'a')
        for val in range(np.shape(data)[0]):
            assert data[val, 0] == np.log(orig_data[val, 0] + 3)
            assert data[val, 1] == orig_data[val, 1]
            assert data[val, 2] == orig_data[val, 2]

    def test_sqrt_transform(self, subject):
        # It should log transform the data
        data = np.array([[1., 2., 3.], [2., 3., 4.], [0., 4., 5.], [4., 5., 6.], [5., 6., 7.], [6., 7., 8.],
                         [7., 8., 9.], [8., 9., 10.], [9., 10., 11.]])
        orig_data = data.copy()

        subject.sqrt_transform(data, 'a')
        for val in range(np.shape(data)[0]):
            assert data[val, 0] == np.sqrt(orig_data[val, 0])
            assert data[val, 1] == orig_data[val, 1]
            assert data[val, 2] == orig_data[val, 2]

        subject.sqrt_transform(data, 'b')
        for val in range(np.shape(data)[0]):
            assert data[val, 0] == np.sqrt(orig_data[val, 0])
            assert data[val, 1] == np.sqrt(orig_data[val, 1])
            assert data[val, 2] == orig_data[val, 2]

        # It should never be able to contain a value < 0
        data = np.array([[0., 0., 0.], [-2., -3., -4.], [3., 4., 5.], [4., 5., 6.], [5., 6., 7.], [6., 7., 8.],
                         [7., 8., 9.], [8., 9., 10.], [9., 10., 11.]])
        orig_data = data.copy()
        subject.sqrt_transform(data, 'a')
        for val in range(np.shape(data)[0]):
            assert data[val, 0] == np.sqrt(orig_data[val, 0] + 3)
            assert data[val, 1] == orig_data[val, 1]
            assert data[val, 2] == orig_data[val, 2]
