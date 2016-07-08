from data_output.plotters.plotter import Plotter
import matplotlib.pyplot as plt

class ActualVsPredictionPlotter(Plotter):

    def plot(self, actual, predicted):
        if predicted is None:
            return False

        fig, ax = plt.subplots()

        # Plot the predicted values against the actual values
        ax.scatter(actual, predicted)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        return plt
