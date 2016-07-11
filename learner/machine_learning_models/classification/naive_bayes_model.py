from machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn.naive_bayes import GaussianNB


class NaiveBayesModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names, model_type='classification')
        self.skmodel = GaussianNB()

    def predict_for_roc(self, x_data):
        # TODO: What is the difference between log_proba and proba?
        return self.skmodel.predict_log_proba(x_data)[:,1]