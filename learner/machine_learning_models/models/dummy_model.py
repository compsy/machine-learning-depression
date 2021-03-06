from learner.machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn.dummy import DummyClassifier


class DummyClassifierModel(MachineLearningModel):

    def __init__(self, x, y, y_names, grid_search, verbosity, **kwargs):
        hyperparameters = {'strategy': 'constant', 'constant': 0}
        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, verbosity=verbosity, pretty_name = 'Constant Dummy', model_type='classification', **kwargs)
        self.skmodel = DummyClassifier(**self.hyperparameters)


class DummyRandomClassifierModel(MachineLearningModel):

    def __init__(self, x, y, y_names, grid_search, verbosity, **kwargs):
        hyperparameters = {'strategy': 'uniform'}
        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, verbosity=verbosity, pretty_name = 'Random Dummy', model_type='classification', **kwargs)
        self.skmodel = DummyClassifier(**self.hyperparameters)
