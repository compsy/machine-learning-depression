import sklearn

from learner.data_output.plotters.plotter import Plotter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from learner.data_output.std_logger import L


class LearningCurvePlotter(Plotter):

    def plot(self, model, ylim=None, cv=10, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 50)):
        """
        Generate a simple plot of the test and traning learning curve.
        from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#example-model-selection-plot-learning-curve-py
        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or models;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : integer, cross-validation generator, optional
            If an integer is passed, it is the number of folds (defaults to 3).
            Specific cross-validation objects can be passed, see
            sklearn.cross_validation module for the list of possible objects

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
        plt.figure()
        plt.title('Learning curves for ' + model.given_name)
        plt.xlabel("Training examples")
        plt.ylabel("Score")

        plot_name = model.given_name
        plot_name = 'learning_' + plot_name.replace(" ", "_")

        L.info('Determining learning curve for ' + model.given_name)
        L.info('Which uses: ' + str(model.skmodel))
        train_sizes, train_scores, test_scores = learning_curve(
            model.skmodel,
            model.x_train,
            model.y_train,
            cv=cv,
            train_sizes=train_sizes,
            scoring=model.scoring(),
            n_jobs=-1,
            verbose=1)
        L.info('Done fitting learning curve')

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        # Create the shading
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r")
        plt.fill_between(
            train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

        # Plot the means of the lines
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        return self.return_file(plt, plot_name)
