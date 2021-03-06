import numpy as np
from learner.data_output.std_logger import L
from learner.data_output.datatool_output import DatatoolOutput

class FeatureSelector():

    def determine_best_variables(self, mlmodel, top=25):
        DatatoolOutput.export('number-covariates-from-feature-selection', top)
        if mlmodel.was_trained:
            coef_ = mlmodel.skmodel.coef_[0]
            assert len(coef_) == len(mlmodel.x_names)
            indices = list(range(len(coef_)))
            return self.return_top(coef_, indices, mlmodel, top)

    def determine_best_variables_elastic_net(self, mlmodel, top=25):
        if mlmodel.was_trained:
            assert len(mlmodel.skmodel.coef_) == len(mlmodel.x_names)
            indices = mlmodel.skmodel.sparse_coef_.indices
            data = mlmodel.skmodel.sparse_coef_.data
            return self.return_top(data, indices, mlmodel, top)

    def return_top(self, coef, indices, mlmodel, top):
        #L.info('The most predictive variables are:')
        zipped = list(zip(coef, indices))
        zipped.sort(reverse=True, key=lambda tup: abs(tup[0]))
        i = 0
        var_names = []
        for coefficient, index in zipped:
            i += 1
            var_name = mlmodel.x_names[index]
            var_names.append([var_name, coefficient])
            #L.info('--> %d\t%0.5f\t%s' % (i, coefficient, var_name))
            if (i >= top): break

        return np.array(var_names)
