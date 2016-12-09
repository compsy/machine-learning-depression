import sklearn

import sys
from learner.data_output.plotters.plotter import Plotter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from learner.data_output.std_logger import L


class DataDensityPlotter(Plotter):

    def plot(self, X, header, sampling_density=1000):
        X_plot = np.linspace(0, sampling_density, sampling_density)[:, np.newaxis]
        bins = np.linspace(0, 100, 100)

        nr_of_variables = np.shape(X)[1]
        cols = int(np.round(np.sqrt(nr_of_variables)))
        rows = int(np.floor(nr_of_variables / cols)) + 1

        plt.yticks([-1, 0, +1])
        fig, ax = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(cols * 5, rows * 5), dpi=72)
        row = 0
        col = 0

        L.info('Plotting distribution (%d cols, %d rows)' % (cols, rows))
        for variable in range(nr_of_variables):
            fig.subplots_adjust(hspace=0.05, wspace=0.05)

            sys.stdout.write('\r\t --> %d/%d' % (variable, nr_of_variables))
            x = X[:, variable]

            unique_entries = len(np.unique(x))
            beg = min(np.unique(x))
            end = max(np.unique(x))
            bins = np.linspace(beg, end, 100)
            if (unique_entries < 10): bins = np.linspace(beg, end, unique_entries + 1)

            # histogram 1
            ax[row, col].hist(x, bins=bins, fc='#AAAAFF', normed=True)
            ax[row, col].set_title(header[variable])

            col += 1
            col = col % cols
            if (col == 0): row += 1

            # kde = KernelDensity(kernel='tophat', bandwidth=0.1).fit(X[:,[variable]])
            #
            # # score_samples() returns the log-likelihood of the samples
            # dens = np.exp(kde.score_samples(X_plot))
            #
            # ax[1, 1].fill(X_plot[:, 0], dens, fc='#AAAAFF')
            # ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")
        L.br()

        # kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
        # x = np.linspace(0, sampling_density, sampling_density)
        # y = kde.sample(sampling_density)[:,0]
        #
        # plt.figure()
        # plt.fill(x, y, 'r')
        # plt.grid(True)

        plot_name = 'density_plot'

        return self.return_file(plt, plot_name)
