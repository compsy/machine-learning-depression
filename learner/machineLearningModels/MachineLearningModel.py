import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split


class MachineLearningModel:

    def __init__(self, data, header, x, y):
        self.data = self.remove_missings(data)
        self.header = header
        self.x_names = x
        self.y_names = y
        self.x = self.split_data(x)
        self.y = self.split_data(y)

    def remove_missings(self, data):
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(data)
        return imp.transform(data)

    def split_data(self, variable_names):
        variable_indices = []
        for name in variable_names:
            variable_indices.append(np.where(self.header == name)[0][0])
        return self.data[:, variable_indices]

    def train_test_data(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.33,
            random_state=42)
        return (x_train, x_test, y_train, y_test)

    def train(self):
        raise NotImplementedError

    def plot(self, predicted):
        if predicted is None:
            return False

        fig, ax = plt.subplots()
        ax.scatter(self.y, predicted)
        ax.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()],
                'k--',
                lw=4)
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
