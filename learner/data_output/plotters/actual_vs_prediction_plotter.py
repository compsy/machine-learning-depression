from data_output.plotters.plotter import Plotter
import matplotlib.pyplot as plt

class ActualVsPredictionPlotter(Plotter):

    def plot(self, model, actual, predicted):
        if predicted is None or actual is None:
            return False

        plot_name = model.given_name
        plot_name = 'actual_vs_prediction' + plot_name.replace(" ", "_")
        print('\t -> Plotting '+plot_name)
        plt.figure()
        plt.title('Act vs Pred: ' + model.given_name)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')

        # Plot the predicted values against the actual values
        plt.scatter(actual, predicted)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)

        return self.return_file(plt, plot_name)
