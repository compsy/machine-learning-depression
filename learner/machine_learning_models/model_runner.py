import numpy as np
from learner.data_output.std_logger import L
from learner.machine_learning_models.models.bagging_model import BaggingClassificationModel, BaggingModel


class ModelRunner:
    """
    Class that automatically fabricates and runs the algorithms provided to it.
    Parameters
    ----------
    algorithms : list, the list of algorithms as specified in driver.py
    """

    def __init__(self, algorithms):
        self.algorithms = []
        self.algorithm_options = []
        for algorithm in algorithms:
            self.algorithms.append(algorithm['model'])
            self.algorithm_options.append(algorithm['options'])

    def run_calculations(self, fabricated_models):
        """
        Run calculations should be implemented by a subclass. It is the
        function that actually runs the calculations for each algorithm
        """
        raise NotImplementedError('This function needs to be implemented in the subclass')

    def fabricate_models(self, x_data, y_data, y_names, verbosity):
        """
        Fabricates (that is, creates instances) of the algorithms in
        self.algorithms. Furthermore, it will enable gridsearch and bagging for
        those algorithms if specified.

        :param x_data: the training x_data values
        :param y_data: the training y_data values
        :param y_names: the names (variable names) of the y_data data
        :param verbosity: whether it has to be verbose or not
        :return: a list of fabricated / instances of ML algorithms
        """
        L.info('Fabricating algorithms')
        created_models = []

        for i, algorithm in enumerate(self.algorithms):
            # Get the options for the current algorithm
            has_grid_search = True if ('grid-search' in self.algorithm_options[i]) else False
            is_bagged = 'bagging' in self.algorithm_options[i]

            # Fabricate the internal model
            current_model = algorithm(
                x=x_data, y=y_data, y_names=y_names, bagged=is_bagged, verbosity=verbosity, grid_search=has_grid_search)

            # Hier een check maken om te kijken of het algorithm daadwerkelijk geupdate is met
            # nieuwe hyperparameters?
            if is_bagged:
                current_model.skmodel = ModelRunner.use_bagging(current_model, verbosity)

            created_models.append(current_model)

        L.info('Done fabricating')
        return created_models

    @staticmethod
    def use_bagging(algorithm, verbosity):
        """ Bags the algorithm provided to it """
        current_skmodel = None
        if (algorithm.model_type == 'regression'):
            current_skmodel = BaggingModel
        elif (algorithm.model_type == 'classification'):
            current_skmodel = BaggingClassificationModel

        return current_skmodel.use_bagging(verbosity=verbosity, skmodel=algorithm.skmodel)
