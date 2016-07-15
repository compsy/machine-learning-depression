from data_output.plotters.plotter import Plotter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve, validation_curve


class ValidationCurvePlotter(Plotter):

    def plot(self, model):

        space = np.logspace(-5, 6, 50)
        space = np.linspace(1, 100, 100)
        plot_name = model.given_name
        plot_name = 'validation_' + plot_name.replace(" ", "_")

        print(space)
        print('\t -> Determining validation curve for ' + model.given_name + ', using space: ' + str(space))
        train_scores, valid_scores = validation_curve(estimator=model.skmodel,
                                                      X=model.x_train,
                                                      y=model.y_train,
                                                      param_name=model.variable_to_validate(),
                                                      param_range=space,
                                                      n_jobs=-1,
                                                      verbose=1)

        print('\t -> Validating: ' + model.variable_to_validate())
        plt.figure()
        plt.title('Validation curves for ' + model.given_name)
        plt.xlabel(model.variable_to_validate())
        plt.ylabel("Score")

        plt.xlim(min(space), max(space))

        plt.grid()
        train_scores_mean = np.mean(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)

        plt.plot(space, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(space, valid_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        return self.return_file(plt, plot_name)