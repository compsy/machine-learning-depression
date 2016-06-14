from machineLearningModels.MachineLearningModel import MachineLearningModel
from sklearn import svm
from numpy import logspace

class SupportVectorMachineModel(MachineLearningModel):

    def train(self):
	rbf_grid = {'kernel' : ['rbf'], 
		    'C' : [1, 10, 100, 1000], 
		    'epsilon' : logspace(0, 1, 5),
		    'gamma' : logspace(0, 1, 5)}

	poly_grid = {'kernel' : ['poly'], 
		     'C' : [1, 10, 100, 1000], 
		     'degree' : [1, 2, 3, 4, 5],
	             'coef0' : logspace(0, 1, 5),
		     'epsilon' : logspace(0, 1, 5),
		     'gamma' : logspace(0, 1, 5)}

	linear_grid = {'kernel' : ['linear'], 
                       'C' : [1, 10, 100, 1000], 
                       'epsilon' : logspace(0, 1, 5)}

	sigmoid_grid = {'kernel' : ['sigmoid'], 
                        'C' : [1, 10, 100, 1000], 
                        'coef0' : logspace(0, 1, 5),
                        'epsilon' : logspace(0, 1, 5),
                        'gamma' : logspace(0, 1, 5)}

	param_grid = [rbf_grid, poly_grid, linear_grid, sigmoid_grid]
	svm = svm.SVR()
	grid_search_out = GridSearchCV(estimator = svm, param_grid = param_grid)	
        best_params = grid_search_out.best_params
        return cross_val_predict(estimator = svm, X = self.x, y = self.y, cv=10, fit_params = best_params)
