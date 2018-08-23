import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from learner.data_output.latex_table_exporter import LatexTableExporter
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

        f1_scores       = [f1_evaluator.evaluate(truth, outcome) for outcome in outcomes]
        accuracy_scores = [accuracy_evaluator.evaluate(truth, outcome) for outcome in outcomes]
        gmean_scores    = [gmean_evaluator.evaluate(truth, outcome) for outcome in outcomes]
        kappa_scores    = [kappa_evaluator.evaluate(truth, outcome) for outcome in outcomes]
        auc_scores      = [auc_evaluator.evaluate(truth, proba) for proba in probas]
        average_scores  = [sum(x) / len(x) for x in zip(accuracy_scores, f1_scores, gmean_scores, kappa_scores, auc_scores)]

        estimator_codes = [estimator[0] for estimator in estimator_names]
        estimator_their_names = [estimator[1] for estimator in estimator_names]

        sort_by = average_scores
        sorted_indices = list(reversed(sorted(range(N), key=lambda k: sort_by[k])))
        # sorted_indices = range(N)

        average_scores        = itemgetter(*sorted_indices)(average_scores)
        f1_scores             = itemgetter(*sorted_indices)(f1_scores)
        accuracy_scores       = itemgetter(*sorted_indices)(accuracy_scores)
        auc_scores            = itemgetter(*sorted_indices)(auc_scores)
        gmean_scores          = itemgetter(*sorted_indices)(gmean_scores)
        kappa_scores          = itemgetter(*sorted_indices)(kappa_scores)

        estimators            = itemgetter(*sorted_indices)(estimator_codes)
        estimator_their_names = itemgetter(*sorted_indices)(estimator_their_names)

        for index, score in enumerate(average_scores):
            DatatoolOutput.export('performance-' + str(index), round(score, 3))

        DatatoolOutput.export('best-performing-algorithm',                estimator_their_names[0])
        DatatoolOutput.export('best-performing-algorithm-average_score',  round(average_scores[0],3))
        DatatoolOutput.export('best-performing-algorithm-accuracy_score', round(accuracy_scores[0],3))
        DatatoolOutput.export('best-performing-algorithm-auc_score',      round(auc_scores[0],3))
        DatatoolOutput.export('best-performing-algorithm-gmean_score',    round(gmean_scores[0],3))
        DatatoolOutput.export('best-performing-algorithm-kappa_score',    round(kappa_scores[0],3))

        width = 0.15
        ind = np.arange(N)

        plot_name = 'z_performance_per_learner_' + output_type
        ax.set_title('')

        # colors=["#332288", "#88CCEE", "#117733", "#DDCC77", "#CC6677"]
        colors=["#332288", "#88CCEE", "#117733", "#DDCC77", "#CC6677","#AA4499"]

        average_plt  = ax.bar(ind+0*width, average_scores, width, color  = colors[0])
        f1_plt       = ax.bar(ind+1*width, f1_scores, width, color       = colors[1])
        accuracy_plt = ax.bar(ind+2*width, accuracy_scores, width, color = colors[2])
        auc_plt      = ax.bar(ind+3*width, auc_scores, width, color      = colors[3])
        gmean_plt    = ax.bar(ind+4*width, gmean_scores, width, color    = colors[4])
        kappa_plt    = ax.bar(ind+5*width, kappa_scores, width, color    = colors[5])

        plots = [average_plt[0], f1_plt[0] , accuracy_plt[0], auc_plt[0], gmean_plt[0], kappa_plt[0]]
        labels = ['Average', 'F1-score', 'Accuracy', 'AUC', 'Geometric-mean', 'Kappa score']
        data = list(zip(estimator_their_names,
            np.round(average_scores, 3),
            np.round(f1_scores,3),
            np.round(accuracy_scores,3),
            np.round(auc_scores,3),
            np.round(gmean_scores,3),
            np.round(kappa_scores,3)))

        LatexTableExporter.export('exports/z_performance.tex', data, np.append('Algorithm', labels))


        # -1 because of the average score
        DatatoolOutput.export('number-of-performance-measures', DatatoolOutput.number_to_string(len(colors) -1))
        # start from one to skip the average
        labels_down = [(label.lower() if label != 'F1-score' else label) for label in labels]
        DatatoolOutput.export('performance-measures', ", the ".join(labels_down[1:-1]) +", and the "+labels_down[-1])



        # ax.legend((f1_plt[0] , accuracy_plt[0], auc_plt[0]), ('F1-Scores', 'Accuracy', 'AUC'))
        # See https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean-in-matplotlib/39806180#39806180
        # (x0, y0, width, height)
        ax.legend(plots, labels, bbox_to_anchor=(0., -0.2, 1., .102), loc='upper center', ncol=3, mode="expand", borderaxespad=0.)
        ax.set_xticks(ind + 2.5 * width)
        ax.set_xticklabels(estimators)


        # legend = plt.legend(loc="best")
        # legend.remove()
        return self.return_file(plt, plot_name)

