import numpy as np
import pytest
from learner.machineLearningModels import machine_learning_model


class TestMachineLearningModel:

    def test_split_data(self):
        self.data = np.matrix([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                               [7, 8], [8, 9], [9, 10]])
        expected = self.data[:, 0]
        self.x = np.array(['input'])
        self.y = np.array(['output'])
        self.header = np.append(self.x, self.y)
        self.model = machine_learning_model.MachineLearningModel(
            self.data, self.header, self.x, self.y)
        result = self.model.split_data(['input'])
        assert result is not None
        assert np.array_equal(result, expected)

    def test_train(self):
        self.data = np.matrix([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                               [7, 8], [8, 9], [9, 10]])
        expected = self.data[:, 0]
        self.x = np.array(['input'])
        self.y = np.array(['output'])
        self.header = np.append(self.x, self.y)
        self.model = machine_learning_model.MachineLearningModel(
            self.data, self.header, self.x, self.y)
        with pytest.raises(NotImplementedError):
            self.model.train()
