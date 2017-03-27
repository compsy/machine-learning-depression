from sklearn.grid_search import BaseSearchCV, ParameterSampler
from learner.data_output.std_logger import L


class RandomizedSearchMine(BaseSearchCV):

    def __init__(self,
                 estimator,
                 param_distributions,
                 n_iter=10,
                 scoring=None,
                 fit_params=None,
                 n_jobs=1,
                 iid=True,
                 refit=True,
                 cv=None,
                 verbose=0,
                 pre_dispatch='2*n_jobs',
                 random_state=None,
                 error_score='raise'):

        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super(RandomizedSearchMine, self).__init__(
            estimator=estimator,
            scoring=scoring,
            fit_params=fit_params,
            n_jobs=n_jobs,
            iid=iid,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score)

    def fit(self, X, y=None):
        fitted_models = []
        for param_set in self.param_distributions:
            sampled_params = list(ParameterSampler(param_set, self.n_iter, random_state=self.random_state))
            L.info('Running on a param_set with %d iterations' % self.n_iter)
            fitted_model = self._fit(X, y, sampled_params)
            L.info('Finished running on a param_set with %d iterations' % self.n_iter)
            fitted_models.append((fitted_model.best_score_, fitted_model.best_estimator_))
        fitted_models.sort(reverse=True, key=lambda tup: tup[0])
        return fitted_models[0][1]
