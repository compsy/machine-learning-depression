from sklearn.grid_search import BaseSearchCV, ParameterGrid


class GridSearchMine(BaseSearchCV):

    def __init__(self,
                 estimator,
                 param_grid,
                 scoring=None,
                 fit_params=None,
                 n_jobs=1,
                 iid=True,
                 refit=True,
                 cv=None,
                 verbose=0,
                 pre_dispatch='2*n_jobs',
                 error_score='raise'):

        super(GridSearchMine, self).__init__(estimator, scoring, fit_params, n_jobs, iid, refit, cv, verbose,
                                             pre_dispatch, error_score)
        self.param_grid = param_grid

    def fit(self, X, y=None):
        return self._fit(X, y, self.param_grid)
