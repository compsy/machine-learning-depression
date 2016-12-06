import numpy as np
import pytest
from learner.machine_learning_models.grid_search_mine import GridSearchMine



class TestGridSearchMine:
    @pytest.fixture()
    def subject(self):
        estimator = 'estimator'
        param_grid = [123]
        subject = GridSearchMine(estimator, param_grid)
        return subject

    def test_train(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        x_names = np.array(['input'])
        y_names = np.array(['output'])
        model = machine_learning_model.MachineLearningModel(x, y, x_names, y_names)
        with pytest.raises(NotImplementedError):
            model.train()
