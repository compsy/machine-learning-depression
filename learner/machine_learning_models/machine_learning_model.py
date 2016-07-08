from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split


class MachineLearningModel:

    def __init__(self, x, y, x_names, y_names):
        self.skmodel = None
        self.x = x
        self.y = y
        self.x_names = x_names
        self.y_names = y_names
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_data()

    def remove_missings(self, data):
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(data)
        return imp.transform(data)

    def train_test_data(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.20, random_state=42)
        return (x_train, x_test, y_train, y_test)

    def print_accuracy(self):
        scores = self.validate()
        print("%s - Accuracy: %0.2f (+/- %0.2f)" % (self.given_name(), scores.mean(), scores.std() * 2))

    def validate(self):
        self.train()
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validated:
        return cross_val_score(self.skmodel, self.x_train, self.y_train, cv=8)

    def cv_predict(self):
        self.train()
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validated:
        return cross_val_predict(self.skmodel, self.x_train, self.y_train, cv=8)

    # Delegate default scikit learn functions
    def predict(self, *args, **kwargs):
        return self.skmodel.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.skmodel.fit(*args, **kwargs)

    def score(self, *args, **kwargs):
        return self.skmodel.score(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        return self.skmodel.get_params(*args, **kwargs)

    @property
    def given_name(self):
        return type(self).__name__


    def train(self):
        if (self.skmodel is not None):
            return self

        self.skmodel = self.fit(self.x_train, self.y_train)
        return self
