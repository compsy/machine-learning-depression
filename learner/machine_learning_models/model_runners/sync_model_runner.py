from learner.data_output.std_logger import L
from learner.machine_learning_models.model_runner import ModelRunner


class SyncModelRunner(ModelRunner):
    """ The syncmodelrunner calculates / fits a machine learning model for each of the machine
    learning algorithms defined.
    Parameters
    ----------
    algorithms : list, list of machine learning algorithms to use
    """

    def __init__(self, algorithms):
        super().__init__(algorithms)

    def run_calculations(self, fabricated_models):
        """
        Calculates all algorithms in a synchronous fashion. We expect each of the algorithms to do
        their own parallelization.
        """
        result = True
        for model in fabricated_models:
            L.info('Training from syncmodelrunner')
            result = model.train()

        # Convert the outcome to a boolean
        result = result != False
        return (result, fabricated_models)
