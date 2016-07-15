import sklearn
from sklearn.cross_validation import StratifiedKFold

from data_output.plotters.plotter import Plotter
from scipy import interp

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class RocCurvePlotter(Plotter):

    def plot(self, models, cv=6):
        plt.figure(figsize=(12, 12), dpi=72)
        plt.title('Receiver Operator Curves')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plot_name = 'roc_curves'
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random guess')

        for model in models:
            print('\t -> Determining ROC curve for ' + model.given_name)
            print('\t -> Which uses: ' + str(model.skmodel))

            probas_ = model.predict_for_roc(model.x_test)

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(model.y_test, probas_[:])

            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f) for %s' % (roc_auc, model.given_name))

        print('\t -> Done fitting learning curve')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        plt.grid()

        plt.legend(loc="best")
        return self.return_file(plt, plot_name)
