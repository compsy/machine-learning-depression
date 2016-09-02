from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

from machine_learning_evaluation.explained_variance_evaluation import ExplainedVarianceEvaluation
from machine_learning_evaluation.f1_evaluation import F1Evaluation
from machine_learning_evaluation.mse_evaluation import MseEvaluation, RootMseEvaluation
from data_output.std_logger import L
from machine_learning_evaluation.variance_evaluation import VarianceEvaluation
from machine_learning_models.distributed_grid_search import DistributedGridSearch
from machine_learning_models.distributed_random_grid_search import DistributedRandomGridSearch
from machine_learning_models.randomized_search_mine import RandomizedSearchMine


class MachineLearningModel:

    def __init__(self, x, y, x_names, y_names, model_type='models', verbosity=0, hpc=False, n_iter=1000):
        self.x = x
        self.y = y
        self.x_names = x_names
        self.y_names = y_names
        self.grid_model = None
        self.skmodel = None
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_data()
        self.model_type = model_type
        self.was_trained = False
        self.hpc = hpc
        self.evaluations = [VarianceEvaluation(), F1Evaluation(), MseEvaluation(), ExplainedVarianceEvaluation(),
                            RootMseEvaluation()]

        self.grid_search_type = 'random'
        self.n_iter = n_iter

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
        L.info("%s - Accuracy: %0.2f (+/- %0.2f)" % (self.given_name, scores.mean(), scores.std() * 2))

    def print_evaluation(self):
        L.br()
        L.info('SCORES OF MODEL: ' + self.given_name)
        L.info('---------------------------------------------------------')
        self.print_accuracy()
        prediction = self.skmodel.predict(self.x_test)
        for evaluator in self.evaluations:
            if evaluator.problem_type == self.model_type:
                evaluator.print_evaluation(self, self.y_test, prediction)
        L.info('---------------------------------------------------------')

    def cv_score(self):
        return cross_val_score(self.skmodel, self.x_test, self.y_test, cv=10)

    def train(self):
        if (self.was_trained):
            return True

        if (self.skmodel is None):
            raise NotImplementedError('Skmodel is none!')

        L.info('Training ' + self.given_name)
        if self.grid_model is not None:
            result = self.grid_model.fit(X=self.x_train, y=self.y_train)
        else:
            result = self.skmodel.fit(X=self.x_train, y=self.y_train)

        # This check is needed whenever we run using MPI
        if result != False: self.skmodel = result
        self.was_trained = True

        if isinstance(self.skmodel, GridSearchCV):
            self.skmodel = self.skmodel.best_estimator_

        L.info('Fitted ' + self.given_name)
        return result

    def cv_predict(self):
        self.skmodel.fit(self.x_train, self.y_train)
        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validated:
        return cross_val_predict(self.skmodel, X=self.x_train, y=self.y_train, cv=10)

    def scoring(self):
        if (self.model_type == 'models'):
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

    def grid_search(self, exhaustive_grid, random_grid):
        if self.hpc:
            if (self.grid_search_type == 'exhaustive'):
                self.grid_model = DistributedGridSearch(ml_model=self, estimator=self.skmodel,
                                                        param_grid=exhaustive_grid,
                                                        cv=10)
                return self.grid_model
            elif (self.grid_search_type == 'random'):
                self.grid_model = DistributedRandomGridSearch(ml_model=self, estimator=self.skmodel,
                                                        param_grid=random_grid,
                                                        cv=10, n_iter=self.n_iter)
                return self.grid_model
        else:
            if (self.grid_search_type == 'exhaustive'):
                self.skmodel = GridSearchCV(estimator=self.skmodel, param_grid=exhaustive_grid,
                                            n_jobs=-1, verbose=1, cv=10)
                return self.skmodel
            elif (self.grid_search_type == 'random'):
                self.skmodel = RandomizedSearchMine(estimator=self.skmodel, param_distributions=random_grid,
                                                  n_jobs=-1, verbose=1, cv=10, n_iter=self.n_iter)
                return self.skmodel

        raise NotImplementedError('Gridsearch type: ' + self.grid_search_type + ' not implemented')

    ## Override
    def predict_for_roc(self, x_data):
        return self.skmodel.predict_proba(x_data)[:, 1]
