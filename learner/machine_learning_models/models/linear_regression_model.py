from machine_learning_models.machine_learning_model import MachineLearningModel

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from machine_learning_models.models.boosting_model import BoostingClassificationModel


class LinearRegressionModel(MachineLearningModel):
    MAX_ITERATIONS = 10000

    def __init__(self, x, y, x_names, y_names, verbosity):
        super().__init__(x, y, x_names, y_names, model_type='regression')
        self.skmodel = linear_model.LassoCV(eps=1e-2,
                                            n_alphas=300,
                                            fit_intercept=True,
                                            normalize=True,
                                            copy_X=False,
                                            max_iter=self.MAX_ITERATIONS,
                                            verbose=verbosity,
                                            n_jobs=-1)


class LogisticRegressionModel(BoostingClassificationModel):

    def __init__(self, x, y, x_names, y_names, verbosity):
        self.skmodel = LogisticRegression(penalty='l2',
                                          C=0.1,
                                          verbose=verbosity,
                                          random_state=42,
                                          tol=0.000001,
                                          max_iter=100000)

        linear_grid = {'penalty': ['l2'], 'C': [0.1]}
        param_grid = [linear_grid]
        self.skmodel = GridSearchCV(estimator=self.skmodel, param_grid=param_grid, n_jobs=-1, verbose=1)
        super().__init__(x, y, x_names, y_names, verbosity=verbosity, model_type='classification')

    def predict_for_roc(self, x_data):
        return self.skmodel.decision_function(x_data)
