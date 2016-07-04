from data_output.plotters.plotter import Plotter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve, validation_curve


class ValidationCurvePlotter(Plotter):

    def plot(self, model, variable_to_validate='max_iter'):
        print('\t -> Determining validation curve for ' + model.given_name)

        space = np.linspace(10, 50, 40)
        space = [int(spacenow) for spacenow in space]
        print(space)
        train_scores, valid_scores = validation_curve(model.skmodel, model.x, model.y, variable_to_validate, space, n_jobs=1)

        plt.figure()
        plt.title('Validation curves for ' + model.given_name)
        plt.xlabel("examples")
        plt.ylabel("Score")

        plt.xlim(min(space), max(space))

        plt.grid()
        train_scores_mean = np.mean(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)

        plt.plot(space, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(space, valid_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        return plt

