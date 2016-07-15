from data_output.plotters.plotter import Plotter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np


class ConfusionMatrixPlotter(Plotter):

    def plot(self, model, actual, predicted):
        if predicted is None or actual is None:
            return False

        cm = confusion_matrix(actual, predicted)

        cmap = plt.cm.Blues
        np.set_printoptions(precision=2)

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 15), dpi=72)

        plot_name = model.given_name
        plot_name = 'confusion_matrix_' + plot_name.replace(" ", "_")
        print('\t -> Plotting ' + plot_name)

        ax[0, 0].imshow(cm, interpolation='nearest', cmap=cmap)
        ax[0, 0].colorbar()
        ax[0, 0].set_title('Confusion matrix: ' + model.given_name)
        ax[0, 0].ylabel('True label')
        ax[0, 0].xlabel('Predicted label')
        ax[0, 0].tight_layout()

        ax[0, 0].xticks(2, np.array('yes', 'no'), rotation=45)
        ax[0, 0].yticks(2, np.array('yes', 'no'))

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        ax[0, 1].imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        ax[0, 1].colorbar()
        ax[0, 1].set_title('Normalized confusion matrix: ' + model.given_name)
        ax[0, 1].ylabel('True label')
        ax[0, 1].xlabel('Predicted label')
        ax[0, 1].tight_layout()

        ax[0, 1].xticks(2, np.array('yes', 'no'), rotation=45)
        ax[0, 1].yticks(2, np.array('yes', 'no'))

        return self.return_file(plt, plot_name)
