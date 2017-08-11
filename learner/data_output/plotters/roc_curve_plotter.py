import sklearn
import math
import numpy as np
import matplotlib.pyplot as plt
from learner.data_output.plotters.plotter import Plotter
from learner.data_output.std_logger import L
from sklearn.metrics import auc, roc_curve


class RocCurvePlotter(Plotter):

    def plot(self, models, output_type, estimator_names, invert=False):
        estimators = [estimator[0] for estimator in estimator_names]

        N = len(models)
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1.0, N))

        plt.figure(figsize=(40, 25), dpi=172)
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        plt.title('Receiver Operator Curves')
        if invert:
            plt.xlabel('False Negative Rate')
            plt.ylabel('True Negative Rate')
        else:
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

        plot_name = 'z_roc_curves_' + output_type
        # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random guess')

        for estimator, model, color in zip(estimators, models, colors):
            probas_ = model.predict_for_roc(model.x)
            pos_label = 1
            if invert:
                pos_label = 0
                probas_ = 1 - probas_

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(model.y, probas_[:], pos_label = pos_label)

            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label=estimator, color = color)

        L.info('Done fitting learning curve')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        # plt.grid()
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.legend(bbox_to_anchor=(0., -0.22, 1., .102), loc='upper center', ncol=math.ceil(N/2), mode="expand", borderaxespad=0.)

        # legend = plt.legend(loc="best")
        # legend.remove()
        return self.return_file(plt, plot_name)
