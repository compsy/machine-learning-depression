import time
import uuid
import numpy as np

from sklearn.model_selection import GridSearchCV

from learner.caching.cacher import Cacher
from learner.caching.s3_cacher import S3Cacher
from learner.machine_learning_evaluation.accuracy_evaluation import AccuracyEvaluation
from learner.machine_learning_evaluation.explained_variance_evaluation import ExplainedVarianceEvaluation
from learner.machine_learning_evaluation.f1_evaluation import F1Evaluation
from learner.machine_learning_evaluation.mse_evaluation import MseEvaluation, RootMseEvaluation
from learner.data_output.std_logger import L
from learner.machine_learning_evaluation.variance_evaluation import VarianceEvaluation
from learner.machine_learning_models.randomized_search_mine import RandomizedSearchMine

class MachineLearningModel:
    """
    Superclass for each machine learning model
    """
    def __init__(self, x, y, y_names, hyperparameters, model_type='models', bagged = False, verbosity=0, n_iter=100):
        self.x = x
        self.y = np.ravel(y) #convert the 1d matrix to a vector
        self.x_names = x.columns
        self.y_names = y_names
        self.skmodel = None
        self.cv = 10
        self.bagged = bagged
        self.model_type = model_type
        self.was_trained = False

        # The methods to use for evaluating
        self.evaluations = [
            VarianceEvaluation(),
            F1Evaluation(),
            MseEvaluation(),
            ExplainedVarianceEvaluation(),
            RootMseEvaluation(),
            AccuracyEvaluation()
        ]

        # Setup the cacher
        self.cacher = S3Cacher(directory=self.cache_directory())

        self.grid_search_type = 'random'
        self.calculation_time = -1

        # Initialize the hyperparameters from cache, if available
        self.hyperparameters = self.hot_start(hyperparameters)
        self.n_iter = n_iter

    @staticmethod
    def cache_directory():
        """
        Directory to cache the data to
        """
        return 'cache/mlmodels/'

    def print_accuracy(self):
        """
        Prints the accuracy of a model using crossvalidation on the data set
        """
        scores = self.skmodel.score(self.x, self.y)
        L.info("%s - Accuracy: %0.2f (+/- %0.2f)" % (self.given_name, scores.mean(), scores.std() * 2))

    def print_evaluation(self):
        """
        Prints a short evaluation of each of the algorithms
        """
        L.br()
        L.info('SCORES OF MODEL: ' + self.given_name)
        L.info('---------------------------------------------------------')
        prediction = self.skmodel.predict(self.x)
        L.info('Performance on (%d,%d)' % np.shape(self.x))
        for evaluator in self.evaluations:
            if evaluator.problem_type == self.model_type:
                evaluator.print_evaluation(self, self.y, prediction)

        L.info(self.skmodel.get_params())
        L.info('It took ' + str(self.get_calculation_time) + ' to calculate this model')
        L.info('---------------------------------------------------------')

    def inject_trained_model(self, skmodel):
        """
        THis allows a model to be injected that was trained before. This is
        used when caching a fitted model.
        """
        self.was_trained = True
        self.skmodel = skmodel

    def train(self, cache_result=True):
        """
        Performs the actual training step.
        Parameters
        ----------
        cache_results : boolean, default = True, should we store the data?
        returns: the SKmodel
        """
        if self.was_trained:
            return True

        if self.skmodel is None:
            raise NotImplementedError('Skmodel is none!')

        # Store the time for debugging later
        tic = time.time()

        # Fit the actual model
        L.info('Training ' + self.given_name + ' with data (%d, %d)' % np.shape(self.x))
        self.inject_trained_model(skmodel=self.skmodel.fit(X=self.x, y=self.y))

        # We have to store the model name before we actually set the best estimator
        model_name = self.short_name
        if isinstance(self.skmodel, GridSearchCV):
            self.inject_trained_model(skmodel=self.skmodel.best_estimator_)

        # Calculate the time it took
        self.calculation_time = time.time() - tic

        if cache_result:
            self.cache_model(model_name=model_name)

        L.info('Fitted ' + self.given_name)
        return self.skmodel

    def cache_model(self, model_name=None):
        """
        Caches the current model to a dump file
        Parameters
        ----------
        model_name: string, default = None, the name of the cachefile to use
        """
        model_name = MachineLearningModel.short_name if model_name is None else model_name
        data = {
            'score': self.skmodel.score(self.x, self.y),
            'hyperparameters': self.skmodel.get_params(),
            'skmodel': self.skmodel,
            'calculation_time': self.get_calculation_time,
            'is_bagged': self.is_bagged
        }

        # We add a randid to the end so we can store the same model multiple times
        rand_id = uuid.uuid4()
        cache_name = model_name +'_' + str(rand_id) + '.pkl'
        self.cacher.write_cache(data=data, cache_name=cache_name)

    def variable_to_validate(self):
        """
        This is the variable to validate when running variable validation?
        """
        return 'max_iter'

    def hot_start(self, hyperparameters):
        """
        Function to load the hyperparameters from cache. If no hyperparameter cache exists, it returns the defaults.

        Parameters
        ----------
        hyperparameters: the default hyperparameters
        returns: either the default parameters, or the cached parameters (which
            are by definition better than the default)
        """

        # The name where the cache should be (more or less, without the randomstring)
        cache_name = self.short_name
        needed_fields_in_cache = ['score', 'hyperparameters', 'skmodel', 'calculation_time', 'is_bagged']

        # List all files in the dir, and filter out the hyper parameters for the current one.
        files = self.cacher.files_in_dir()
        files = [file_name for file_name in files if cache_name in file_name]

        self.hot_started = False
        best_score = 0
        for filename in files:
            cached_params = self.cacher.read_cache(filename)
            if Cacher.is_valid_cache(cached_params, needed_fields_in_cache):
                if not (self.is_bagged ^ cached_params['is_bagged']):
                    if cached_params['score'] > best_score:
                        self.hot_started = True
                        self.calculation_time = cached_params['calculation_time']
                        best_score = cached_params['score']
                        hyperparameters = cached_params['hyperparameters']

        # Bag our model again here.
        if self.is_bagged:
            # Only the keys prefixed with base_estimator__ are the keys we want.
            prefix = 'base_estimator__'
            keys = [key for key in hyperparameters.keys() if key.startswith(prefix)]
            hyperparameters = {key[len(prefix):]: hyperparameters[key] for key in keys}

        return hyperparameters

    @property
    def is_hot_started(self):
        """
        Returns a boolean whether the model has been successfully hot started
        """
        return self.hot_started

    @property
    def is_bagged(self):
        """
        Returns a boolean whether the model is bagged
        """
        return self.bagged

    @property
    def get_y(self):
        """
        Returns the dataset (y)
        """
        return self.y

    @property
    def get_x(self):
        """
        Returns the dataset (x)
        """
        return self.x

    @property
    def given_name(self):
        """
        Longer name representing the current object
        """
        bagging_string = '(bagged)' if self.is_bagged else ''
        return type(self).__name__ + " Type: " + type(self.skmodel).__name__ + bagging_string

    @property
    def short_name(self):
        """
        Short name representing the current model
        """
        bagging_string = '_bagged' if self.is_bagged else ''
        return type(self).__name__+ bagging_string# + type(self.skmodel).__name__

    @property
    def get_calculation_time(self):
        """
        Returns the time it took to train the algorithm (-1 if it has not been trained yet)
        """
        return self.calculation_time

    def grid_search(self, exhaustive_grid, random_grid):
        """
        Perform the actual gridsearch on the model. Note that we also refit the best model on the whole dataset.
        """
        if self.grid_search_type == 'random':
            self.skmodel = RandomizedSearchMine(
                estimator=self.skmodel, param_distributions=random_grid, n_jobs=-1, cv=self.cv, n_iter=self.n_iter
            )
            return self.skmodel

        elif self.grid_search_type == 'exhaustive':
            self.skmodel = GridSearchCV(
                estimator=self.skmodel, param_grid=exhaustive_grid, n_jobs=-1, verbose=1, cv=self.cv
            )
            return self.skmodel

        raise NotImplementedError('Gridsearch type: ' + self.grid_search_type + ' not implemented')

    def predict_for_roc(self, x_data):
        """
        The ROC function needs a function to qualify whether an algorithm
        performs well. This is the default. Whenever we dont have this method
        in an estimator, override it in the subclass.
        """
        return self.skmodel.predict_proba(x_data)[:, 1]

