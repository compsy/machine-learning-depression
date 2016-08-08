from machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn import svm
from numpy import logspace
from scipy.stats import expon, halflogistic

class SupportVectorModel(MachineLearningModel):
    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, **kwargs)



class SupportVectorRegressionModel(SupportVectorModel):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, verbosity= verbosity, model_type='regression', **kwargs)
        self.skmodel = svm.SVR(verbose=verbosity)
        # Radial basis function grid
        rbf_grid = {'kernel': ['rbf'],
                    'C': [1, 10, 100, 1000],
                    'epsilon': logspace(0, 1, 5),
                    'gamma': logspace(0, 1, 5)}

        # Polynomial function grid
        poly_grid = {'kernel': ['poly'],
                     'C': [1, 10, 100, 1000],
                     'degree': [2, 3, 4, 5],
                     'coef0': logspace(0, 1, 5),
                     'gamma': logspace(0, 1, 5)}
        # Linear function grid
        linear_grid = {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

        # Sigmoid function grid
        sigmoid_grid = {'kernel': ['sigmoid'],
                        'C': [1, 10, 100, 1000],
                        'coef0': logspace(0, 1, 5),
                        'gamma': logspace(0, 1, 5)}

        self.exhaustive_param_grid = [rbf_grid, poly_grid, linear_grid, sigmoid_grid]
        self.exhaustive_param_grid = [rbf_grid, poly_grid, sigmoid_grid]

        random_rbf_grid = {'kernel': ['rbf'],
                           'C': halflogistic(scale=100),
                           'gamma': halflogistic(scale=.1),
                           'epsilon': halflogistic(scale=.1)}

        random_poly_grid = {'kernel': ['poly'],
                           'C': halflogistic(scale=100),
                           'degree': [2, 3, 4, 5],
                           'gamma': halflogistic(scale=.1),
                           'coef0': halflogistic(scale=.1)}

        random_sigmoid_grid = {'kernel': ['sigmoid'],
                           'C': halflogistic(scale=100),
                           'gamma': halflogistic(scale=.1),
                           'coef0': halflogistic(scale=.1)}


        self.random_param_grid = [random_rbf_grid, random_poly_grid, random_sigmoid_grid]

        self.grid_search(self.exhaustive_param_grid, self.random_param_grid)


class SupportVectorClassificationModel(SupportVectorModel):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, verbosity= verbosity, model_type='classification', **kwargs)
        self.skmodel = svm.SVC(verbose=verbosity, kernel='poly', degree=2, C=600000)
        # Radial basis function grid
        rbf_grid = {'kernel': ['rbf'],
                    'C': [1, 10, 100, 1000],
                    'gamma': logspace(0, 1, 5),
                    'class_weight': ['balanced', None]}

        # Polynomial function grid
        poly_grid = {'kernel': ['poly'],
                     'C': [1, 10, 100, 1000],
                     'degree': [2, 3, 4, 5],
                     'coef0': logspace(0, 1, 5),
                     'gamma': logspace(0, 1, 5)}
        # Linear function grid
        linear_grid = {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

        # Sigmoid function grid
        sigmoid_grid = {'kernel': ['sigmoid'],
                        'C': [1, 10, 100, 1000],
                        'coef0': logspace(0, 1, 5),
                        'gamma': logspace(0, 1, 5)}

        self.exhaustive_param_grid = [rbf_grid, poly_grid, linear_grid, sigmoid_grid]
        self.exhaustive_param_grid = [rbf_grid, poly_grid, sigmoid_grid]

        random_rbf_grid = {'kernel': ['rbf'],
                           'C': halflogistic(scale=100),
                           'gamma': halflogistic(scale=.1),
                           'class_weight': ['balanced', None]}

        random_poly_grid = {'kernel': ['poly'],
                           'C': halflogistic(scale=100),
                           'degree': [2, 3, 4, 5],
                           'gamma': halflogistic(scale=.1),
                           'coef0': halflogistic(scale=.1),
                           'class_weight': ['balanced', None]}

        random_sigmoid_grid = {'kernel': ['sigmoid'],
                           'C': halflogistic(scale=100),
                           'gamma': halflogistic(scale=.1),
                           'coef0': halflogistic(scale=.1),
                           'class_weight': ['balanced', None]}


        self.random_param_grid = [random_rbf_grid, random_poly_grid, random_sigmoid_grid]
        self.grid_search(self.exhaustive_param_grid, self.random_param_grid)

    def variable_to_validate(self):
        return 'degree'
