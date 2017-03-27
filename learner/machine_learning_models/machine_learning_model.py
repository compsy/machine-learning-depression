from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
import numpy as np

from learner.caching.object_cacher import ObjectCacher
from learner.caching.s3_cacher import S3Cacher
from learner.machine_learning_evaluation.accuracy_evaluation import AccuracyEvaluation
from learner.machine_learning_evaluation.explained_variance_evaluation import ExplainedVarianceEvaluation
from learner.machine_learning_evaluation.f1_evaluation import F1Evaluation
from learner.machine_learning_evaluation.mse_evaluation import MseEvaluation, RootMseEvaluation
from learner.data_output.std_logger import L
from learner.machine_learning_evaluation.variance_evaluation import VarianceEvaluation
from learner.machine_learning_models.distributed_grid_search import DistributedGridSearch
from learner.machine_learning_models.distributed_random_grid_search import DistributedRandomGridSearch
from learner.machine_learning_models.randomized_search_mine import RandomizedSearchMine
import uuid

class MachineLearningModel:
    def __init__(self, x, y, x_names, y_names, hyperparameters, model_type='models', verbosity=0, hpc=False, n_iter=100):
        self.x = x
        self.y = y
        self.x_names = x_names
        self.y_names = y_names
        self.grid_model = None
        self.skmodel = None
        self.test_size = 0.1
        self.cv = 10
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_data()
        self.model_type = model_type
        self.was_trained = False
        self.hpc = hpc
        self.evaluations = [
            VarianceEvaluation(), F1Evaluation(), MseEvaluation(), ExplainedVarianceEvaluation(), RootMseEvaluation(),
            AccuracyEvaluation()
        ]

        # Setup the cacher
        self.cacher = S3Cacher(directory=self.cache_directory())

        self.grid_search_type = 'random'

        # Initialize the hyperparameters from cache, if available
        self.hyperparameters = self.hot_start(hyperparameters)

        self.n_iter = n_iter

    @staticmethod
    def cache_directory():
        return 'cache/mlmodels/'

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

    def inject_trained_model(self, skmodel):
        self.was_trained = True
        self.skmodel = skmodel

    def train(self, cache_result = True):
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

        # We have to store the model name before we actually set the best estimator
        model_name = self.model_cache_name
        if isinstance(self.skmodel, GridSearchCV):
            self.skmodel = self.skmodel.best_estimator_

        if cache_result: self.cache_model(model_name=model_name)

        L.info('Fitted ' + self.given_name)
        return result

    def cache_model(self, model_name=None):
        model_name = MachineLearningModel.model_cache_name if model_name is None else model_name
        data = {
            'score': self.skmodel.score(self.x_test, self.y_test),
            'hyperparameters': self.skmodel.get_params(),
            'skmodel': self.skmodel,
        }
        rand_id = uuid.uuid4()
        cache_name = model_name +'_' + str(rand_id) + '.pkl'
        self.cacher.write_cache(data=data, cache_name=cache_name)

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
        """
        Function to load the hyperparameters from cache. If no hyperparameter cache exists, it returns the defaults.
        :param hyperparameters: the default hyperparameters
        :return: either the default parameters, or the cached parameters (which are by definition better than the default)
        """
        cache_name = self.model_cache_name

        files = self.cacher.files_in_dir()

        # Only use the hyperparameters of the for the present model
        files = list(filter(lambda x: cache_name in x, files))

        best_score = 0
        for filename in files:
            cached_params = self.cacher.read_cache(filename)
            if cached_params['score'] > best_score:
                best_score = cached_params['score']
                hyperparameters = cached_params['hyperparameters']

        return hyperparameters

    @property
    def given_name(self):
        return type(self).__name__ + " Type: " + type(self.skmodel).__name__

    @property
    def short_name(self):
        return type(self).__name__ + type(self.skmodel).__name__

    @property
    def model_cache_name(self):
        return self.short_name + '_hyperparameters'

    def grid_search(self, exhaustive_grid, random_grid):
        if (self.grid_search_type == 'random'):
            self.skmodel = RandomizedSearchMine(
                estimator=self.skmodel, param_distributions=random_grid, n_jobs=-1, cv=self.cv, n_iter=self.n_iter)
            return self.skmodel

        elif (self.grid_search_type == 'exhaustive'):
            if self.hpc:
                self.grid_model = DistributedGridSearch(
                    ml_model=self, estimator=self.skmodel, param_grid=exhaustive_grid, cv=self.cv)
                return self.grid_model

            else:
                self.skmodel = GridSearchCV(
                    estimator=self.skmodel, param_grid=exhaustive_grid, n_jobs=-1, verbose=1, cv=self.cv)
                return self.skmodel

        raise NotImplementedError('Gridsearch type: ' + self.grid_search_type + ' not implemented')

    ## Override
    def predict_for_roc(self, x_data):
        return self.skmodel.predict_proba(x_data)[:, 1]
