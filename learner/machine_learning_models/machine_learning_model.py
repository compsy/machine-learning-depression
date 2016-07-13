from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from learner.machine_learning_evaluation.f1_evaluation import F1Evaluation
from learner.machine_learning_evaluation.mse_evaluation import MseEvaluation


class MachineLearningModel:

    def __init__(self, x, y, x_names, y_names, model_type='regression'):
        self.skmodel = None
        self.x = x
        self.y = y
        self.x_names = x_names
        self.y_names = y_names
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_data()
        self.model_type = model_type
        self.was_trained = False
        self.evaluations = [F1Evaluation(), MseEvaluation()]

    def remove_missings(self, data):
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(data)
        return imp.transform(data)

    def train_test_data(self):
        """
        Splits dataset up into train and test set
        """
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.20, random_state=42)
        return (x_train, x_test, y_train, y_test)

    def print_accuracy(self):
        """
        Prints the accuracy of a model using crossvalidation on the test set
        """
        scores = self.skmodel.score(self.x_test, self.y_test)
        print("\t -> %s - Accuracy: %0.2f (+/- %0.2f)" % (self.given_name, scores.mean(), scores.std() * 2))

    def print_evaluation(self):
        print('\t SCORES OF MODEL: ' + self.given_name)
        print('\t ---------------------------------------------------------')
        self.print_accuracy()
        prediction = self.skmodel.predict(self.x_test)
        for evaluator in self.evaluations:
            if (evaluator.problem_type == self.model_type):
                evaluator.print_evaluation(self, self.y_test, prediction)
        print('\t ---------------------------------------------------------')
        print('')

    def cv_score(self):
        return cross_val_score(self.skmodel, self.x_test, self.y_test, cv=3)

    def train(self):
        if (self.was_trained):
            return True

        if (self.skmodel is None):
            raise NotImplementedError

        self.was_trained = True
        self.skmodel = self.skmodel.fit(X=self.x_train, y=self.y_train)

    def cv_predict(self):
        self.skmodel.fit(self.x_train, self.y_train)
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validated:
        return cross_val_predict(self.skmodel, X=self.x_train, y=self.y_train, cv=8)

    def scoring(self):
        if (self.model_type == 'regression'):
            return 'mean_squared_error'
        elif (self.model_type == 'classification'):
            return 'accuracy'
        else:
            raise NotImplementedError('Type: ' + self.type + ' not implented')

    def variable_to_validate(self):
        return 'max_iter'

    @property
    def given_name(self):
        return type(self).__name__
