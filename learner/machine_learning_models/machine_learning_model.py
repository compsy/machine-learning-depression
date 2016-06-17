from sklearn.cross_validation import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split


class MachineLearningModel:

    def __init__(self, x, y, x_names, y_names):
        self.skmodel = None
        self.x = x
        self.y = y
        self.x_names = x_names
        self.y_names = y_names
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_data(
        )

    def remove_missings(self, data):
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(data)
        return imp.transform(data)

    def train_test_data(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=0.33,
                                                            random_state=42)
        return (x_train, x_test, y_train, y_test)

    def print_accuracy(self):
        scores = self.validate()
        print("%s - Accuracy: %0.2f (+/- %0.2f)" %
              (self.given_name(), scores.mean(), scores.std() * 2))

    def validate(self):
        self.train()
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validated:
        return cross_val_score(self.skmodel, self.x_train, self.y_train, cv=8)

    def predict(self):
        self.train()
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validated:
        return cross_val_predict(self.skmodel, self.x_train, self.y_train, cv=8)


    def plot(self, actual, predicted):
        if predicted is None:
            return False

        fig, ax = plt.subplots()

        # Plot the predicted values against the actual values
        ax.scatter(actual, predicted)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
                'k--',
                lw=4)
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()

    def given_name(self):
        return type(self).__name__

    def train(self):
        raise NotImplementedError
