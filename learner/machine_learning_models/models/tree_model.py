from scipy.stats import halflogistic, randint, uniform
from sklearn import tree
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
import pydotplus

from learner.machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from learner.machine_learning_models.models.boosting_model import BoostingClassificationModel
from scipy.stats import expon


class RegressionTreeModel(MachineLearningModel):

    def __init__(self, x, y, y_names, grid_search, verbosity, **kwargs):
        hyperparameters = {'max_depth': 5}
        super().__init__(x, y, y_names, hyperparameters=hyperparameters, pretty_name = 'Regression Tree', model_type='regression', **kwargs)

        self.skmodel = DecisionTreeRegressor(**self.hyperparameters)

        if grid_search:
            parameter_grid = {
                'max_depth': np.logspace(0, 3, 15),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            random_parameter_grid = {'max_depth': logser(p=.99).rvs(), 'max_features': ['auto', 'sqrt', 'log2', None]}
            self.grid_search([parameter_grid], [random_parameter_grid])


class ClassificationTreeModel(MachineLearningModel):

    def __init__(self, x, y, y_names, grid_search, verbosity, **kwargs):
        hyperparameters = {
            'min_samples_split': 2,
            'max_features': 'auto',
            'criterion': 'gini',
            'presort': False,
            'random_state': None,
            'min_weight_fraction_leaf': 0.0,
            'class_weight': None,
            'splitter': 'best',
            'min_samples_leaf': 1,
            'max_depth': 187.62645773985275,
            'max_leaf_nodes': None
        }
        super().__init__(
            x, y, y_names, hyperparameters=hyperparameters, pretty_name = 'Decision Tree', model_type='classification', verbosity=verbosity, **kwargs)

        self.skmodel = DecisionTreeClassifier(**self.hyperparameters)

        if grid_search:
            parameter_grid = {
                'n_estimators': randint(1, 10),
                'max_depth': np.logspace(0, 2, 20),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            random_parameter_grid = {
                # The minimum number of samples required to split an internal node. When used as float, it is considered a percentage ceil(min_samples_split * n_samples)
                'min_samples_split': uniform(0.01, 0.4),
                # The minimum number of samples required to be at a leaf node. If int, then consider min_samples_leaf as the minimum number.
                'min_samples_leaf': randint(1, 100),
                # The maximum depth of the tree.
                'max_depth': randint(1, 400),
                # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
                'min_weight_fraction_leaf': uniform(0,0.5),
                # The number of features to consider when looking for the best split
                'max_features': ['auto', 'sqrt', 'log2', None],
                # The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
                'criterion': ['gini', 'entropy']

            }
            self.grid_search([parameter_grid], [random_parameter_grid])
