from learner.data_output.plotters.plotter import Plotter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score, precision_score

import numpy as np
from learner.data_output.std_logger import L


class ConfusionMatrixPlotter(Plotter):

    def plot(self, models, output_type, estimator_names):

        estimators = [estimator[0] for estimator in estimator_names]
        for estimator, model in zip(estimator_names, models):
            # Get the true and predicted data
            y_pred = model.skmodel.predict(model.get_x)
            y_true = model.get_y

            name = estimator[0]
            cmap = plt.cm.Blues
            # cmap = plt.cm.afmhot
            # cmap = plt.cm.rainbow

            np.set_printoptions(precision=2)

            plt.figure(figsize=(15, 15), dpi=172)
            fig = plt.figure()
            # [left, bottom, width, height] 
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            tick_marks = np.arange(2)

            plt.setp(ax, xticks=tick_marks, xticklabels=['Yes', 'No'], yticks=tick_marks, yticklabels=['Yes', 'No'])

            plot_name = model.given_name
            plot_name = 'z_confusion_matrix_' + output_type + '_' + estimator[0]

            confusion_matrix_calculated = confusion_matrix(y_true, y_pred)

            # im = ax[0].imshow(confusion_matrix_calculated, interpolation='nearest', cmap=cmap)

            # ax[0].set_title('CFM:' + model.given_name)
            # ax[0].set_ylabel('True label')
            # ax[0].set_xlabel('y_pred label')
            # fig.colorbar(im)

            cm_normalized = confusion_matrix_calculated.astype('float') / confusion_matrix_calculated.sum(axis=1)[:, np.newaxis]

            im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
            ax.set_title('')
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')

            fig.colorbar(im)
            # fig.tight_layout()

            self.return_file(plt, plot_name)
