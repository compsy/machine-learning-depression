import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from learner.data_output.plotters.plotter import Plotter
from learner.data_output.std_logger import L
from learner.data_output.datatool_output import DatatoolOutput
from learner.machine_learning_evaluation.accuracy_evaluation import AccuracyEvaluation
from learner.machine_learning_evaluation.f1_evaluation import F1Evaluation
from learner.machine_learning_evaluation.auc_evaluation import AucEvaluation
from learner.machine_learning_evaluation.geometric_mean_evaluation import GeometricMeanEvaluation
from learner.machine_learning_evaluation.kappa_evaluation import KappaEvaluation


class PerformancePerLearnerPlotter(Plotter):

    def plot(self, models, output_type, estimator_names):
        # Create the evaluatores
        accuracy_evaluator = AccuracyEvaluation()
        f1_evaluator = F1Evaluation()
        auc_evaluator = AucEvaluation()
        gmean_evaluator = GeometricMeanEvaluation()
        kappa_evaluator = KappaEvaluation()

        plt.figure(figsize=(40, 25), dpi=172)
        fig = plt.figure()
        # [left, bottom, width, height] 
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])

        N = len(models)

        outcomes = [model.predict(model.x) for model in models]
        probas = [model.predict_for_roc(model.x) for model in models]
        truth = models[0].y

        accuracy_scores = [accuracy_evaluator.evaluate(truth, outcome) for outcome in outcomes]
        f1_scores = [f1_evaluator.evaluate(truth, outcome) for outcome in outcomes]
        gmean_scores = [gmean_evaluator.evaluate(truth, outcome) for outcome in outcomes]
        kappa_scores = [kappa_evaluator.evaluate(truth, outcome) for outcome in outcomes]
        auc_scores = [auc_evaluator.evaluate(truth, proba) for proba in probas]

        estimators = [estimator[0] for estimator in estimator_names]

        sort_by = accuracy_scores
        sorted_indices = list(reversed(sorted(range(N), key=lambda k: sort_by[k])))

        accuracy_scores = itemgetter(*sorted_indices)(accuracy_scores)
        f1_scores =  itemgetter(*sorted_indices)(f1_scores)
        auc_scores = itemgetter(*sorted_indices)(auc_scores)
        gmean_scores = itemgetter(*sorted_indices)(gmean_scores)
        kappa_scores = itemgetter(*sorted_indices)(kappa_scores)
        estimators = itemgetter(*sorted_indices)(estimators)

        width = 0.15
        ind = np.arange(N)

        plot_name = 'z_performance_per_learner_' + output_type
        ax.set_title('Performance measures for each estimator')

        colors=["#332288", "#88CCEE", "#117733", "#DDCC77", "#CC6677"]
        #tol6qualitative=c("#332288", "#88CCEE", "#117733", "#DDCC77", "#CC6677","#AA4499")

        f1_plt = ax.bar(ind, f1_scores, width, color=colors[0])
        accuracy_plt = ax.bar(ind+width, accuracy_scores, width, color=colors[1])
        auc_plt = ax.bar(ind+2*width, auc_scores, width, color=colors[2])
        gmean_plt = ax.bar(ind+3*width, gmean_scores, width, color=colors[3])
        kappa_plt = ax.bar(ind+4*width, kappa_scores, width, color=colors[4])

        plots = [f1_plt[0] , accuracy_plt[0], auc_plt[0], gmean_plt[0], kappa_plt[0]]
        labels = ['F1-Score', 'Accuracy', 'AUC', 'Geometric-mean', 'Kappa']
        DatatoolOutput.export('number-of-performance-measures', len(colors))
        DatatoolOutput.export('performance-measures', ", ".join(labels[:-1]) +", and "+labels[-1])

        # ax.legend((f1_plt[0] , accuracy_plt[0], auc_plt[0]), ('F1-Scores', 'Accuracy', 'AUC'))
        # See https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean-in-matplotlib/39806180#39806180
        # (x0, y0, width, height)
        ax.legend(plots, labels, bbox_to_anchor=(0., -0.2, 1., .102), loc='upper center', ncol=3, mode="expand", borderaxespad=0.)
        ax.set_xticks(ind + 2.5 * width)
        ax.set_xticklabels(estimators)


        # legend = plt.legend(loc="best")
        # legend.remove()
        return self.return_file(plt, plot_name)

