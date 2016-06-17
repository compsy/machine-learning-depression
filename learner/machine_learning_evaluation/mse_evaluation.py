from sklearn.metrics import mean_squared_error


class MseEvaluation:

    @staticmethod
    def evaluate(self, model, x, y):
        y_predicted = []
        for value in y:
            y_predicted.append(model.predict(y))

        return mean_squared_error(y, y_predicted)
