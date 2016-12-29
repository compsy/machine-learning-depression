from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
import numpy as np

from learner.caching.object_cacher import ObjectCacher
from learner.machine_learning_evaluation.explained_variance_evaluation import ExplainedVarianceEvaluation
from learner.machine_learning_evaluation.f1_evaluation import F1Evaluation
from learner.machine_learning_evaluation.mse_evaluation import MseEvaluation, RootMseEvaluation
from learner.data_output.std_logger import L
from learner.machine_learning_evaluation.variance_evaluation import VarianceEvaluation
from learner.machine_learning_models.distributed_grid_search import DistributedGridSearch
from learner.machine_learning_models.distributed_random_grid_search import DistributedRandomGridSearch
from learner.machine_learning_models.randomized_search_mine import RandomizedSearchMine


class MachineLearningModel:

    def __init__(self, x, y, x_names, y_names, hyperparameters, model_type='models', verbosity=0, hpc=False, n_iter=10):
        self.x = x
        self.y = y
        self.x_names = x_names
        self.y_names = y_names
        self.grid_model = None
        self.skmodel = None
        self.test_size = 0.1
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_data()
        self.model_type = model_type
        self.was_trained = False
        self.hpc = hpc
        self.evaluations = [
            VarianceEvaluation(), F1Evaluation(), MseEvaluation(), ExplainedVarianceEvaluation(), RootMseEvaluation()
        ]

        self.cacher = ObjectCacher('cache/mlmodels/')
        self.grid_search_type = 'random'

        # Initialize the hyperparameters from cache, if available
        self.hyperparameters = self.hot_start(hyperparameters)

        self.n_iter = n_iter
        if hpc:
            self.n_iter = 80

    def remove_missings(self, data):
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(data)
        return imp.transform(data)

    def train_test_data(self):
        """
        Splits dataset up into train and test set
        """
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.test_size, random_state=42)
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
        train_prediction = self.skmodel.predict(self.x_train)
        prediction = self.skmodel.predict(self.x_test)
        L.info('Training data performance')
        for evaluator in self.evaluations:
            if evaluator.problem_type == self.model_type:
                evaluator.print_evaluation(self, self.y_train, train_prediction)

        L.info('Test data performance')
        for evaluator in self.evaluations:
            if evaluator.problem_type == self.model_type:
                evaluator.print_evaluation(self, self.y_test, prediction)

        L.info(self.skmodel.get_params())
        L.info('---------------------------------------------------------')

    def train(self):
        if (self.was_trained):
            return True

        if (self.skmodel is None):
            raise NotImplementedError('Skmodel is none!')

        L.info('Training ' + self.given_name + ' with data (%d, %d)' % np.shape(self.x_train))
        if self.grid_model is not None:
            result = self.grid_model.fit(X=self.x_train, y=self.y_train)
        else:
            result = self.skmodel.fit(X=self.x_train, y=self.y_train)

        # This check is needed whenever we run using MPI
        if result != False: self.skmodel = result
        self.was_trained = True

        if isinstance(self.skmodel, GridSearchCV):
            self.skmodel = self.skmodel.best_estimator_

        cache_name = self.short_name + '_hyperparameters.pkl'
        self.cacher.write_cache(data=self.skmodel.get_params(), cache_name=cache_name)

        L.info('Fitted ' + self.given_name)
        return result

    def scoring(self):
        if (self.model_type == 'models'):
            return 'mean_squared_error'
        elif (self.model_type == 'classification'):
            return 'accuracy'
        else:
            raise NotImplementedError('Type: ' + self.model_type + ' not implemented')

    def variable_to_validate(self):
        return 'max_iter'

    def hot_start(self, hyperparameters):
        cache_name = self.short_name + '_hyperparameters.pkl'
        if self.cacher.file_available(cache_name):
            hyperparameters = self.cacher.read_cache(cache_name)
        return hyperparameters

    @property
    def given_name(self):
        return type(self).__name__ + " Type: " + type(self.skmodel).__name__

    @property
    def short_name(self):
        return type(self).__name__ + type(self.skmodel).__name__

    def grid_search(self, exhaustive_grid, random_grid):
        if self.hpc:
            if (self.grid_search_type == 'exhaustive'):
                self.grid_model = DistributedGridSearch(
                    ml_model=self, estimator=self.skmodel, param_grid=exhaustive_grid, cv=10)
                return self.grid_model
            elif (self.grid_search_type == 'random'):
                self.grid_model = DistributedRandomGridSearch(
                    ml_model=self, estimator=self.skmodel, param_grid=random_grid, cv=10, n_iter=self.n_iter)
                return self.grid_model
        else:
            if (self.grid_search_type == 'exhaustive'):
                self.skmodel = GridSearchCV(
                    estimator=self.skmodel, param_grid=exhaustive_grid, n_jobs=-1, verbose=1, cv=10)
                return self.skmodel
            elif (self.grid_search_type == 'random'):
                self.skmodel = RandomizedSearchMine(
                    estimator=self.skmodel,
                    param_distributions=random_grid,
                    n_jobs=-1,
                    verbose=1,
                    cv=10,
                    n_iter=self.n_iter)
                return self.skmodel

        raise NotImplementedError('Gridsearch type: ' + self.grid_search_type + ' not implemented')

    ## Override
    def predict_for_roc(self, x_data):
        return self.skmodel.predict_proba(x_data)[:, 1]
