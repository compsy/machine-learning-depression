from machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn.dummy import DummyClassifier


class DummyClassifierModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names, model_type='classification')
        self.skmodel = DummyClassifier(strategy='constant', constant=0)

    def predict_for_roc(self, x_data):
        return self.skmodel.predict_proba(x_data)[:, 1]


class DummyRandomClassifierModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names, model_type='classification')
        self.skmodel = DummyClassifier(strategy='uniform')

    def predict_for_roc(self, x_data):
        return self.skmodel.predict_proba(x_data)[:, 1]
