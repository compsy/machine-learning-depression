from scipy.stats import halflogistic
from sklearn import tree
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
import pydotplus

from learner.machine_learning_models.machine_learning_model import MachineLearningModel
from sklearn.cross_validation import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from learner.machine_learning_models.models.boosting_model import BoostingClassificationModel
from scipy.stats import expon


class RegressionTreeModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='regression', **kwargs)
        self.skmodel = DecisionTreeRegressor(max_depth=5)

        if grid_search:
            parameter_grid = {
                'max_depth': np.logspace(0, 3, 15),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            random_parameter_grid = {
                'max_depth': halflogistic(scale=100),
                'max_features': ['auto', 'sqrt', 'log2', None]
            }
            self.grid_search([parameter_grid], [random_parameter_grid])


class ClassificationTreeModel(MachineLearningModel):

    def __init__(self, x, y, x_names, y_names, verbosity, grid_search=True, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='classification', verbosity=verbosity, **kwargs)
        self.skmodel = DecisionTreeClassifier(max_depth=5)

        if grid_search:
            parameter_grid = {
                'max_depth': np.logspace(0, 2, 20),
                'max_features': ['auto', 'sqrt', 'log2', None],
            }
            random_parameter_grid = {
                'max_depth': halflogistic(scale=100),
                'max_features': ['auto', 'sqrt', 'log2', None]
            }
            self.grid_search([parameter_grid], [random_parameter_grid])

    # def train(self):
    #     super(ClassificationTreeModel, self).train()
    #
    #     # Plot the tree
    #     dot_data = StringIO()
    #     tree.export_graphviz(self.skmodel, out_file=dot_data)
    #     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #     graph.write_pdf("exports/classification_tree.pdf")
