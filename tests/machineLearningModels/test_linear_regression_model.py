import numpy as np
from learner.machineLearningModels import LinearRegressionModel


class TestLinearRegressionModel:

    def test_train(self):
        data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                         [8, 9], [9, 10]])
        x_names = np.array(['input'])
        y_names = np.array(['output'])
        header = np.append(x_names, y_names)

        # Here we retrieve the fitted model, and see whether it works
        model = LinearRegressionModel.LinearRegressionModel(data, header,
                                                            x_names, y_names)
        model = model.train()

        for (input_data, expected) in data:
            assert abs(expected - model.predict(input_data)[0][0]) <= 0.00000001

        assert abs(11 - model.predict(10)[0][0]) <= 0.00000001
