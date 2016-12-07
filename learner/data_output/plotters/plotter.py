import matplotlib.pyplot as plt


class Plotter:

    def __init__(self, location='exports/'):
        self.location = location

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def return_file(self, plot, file_name):
        self.save_file(plot, file_name)

    def save_file(self, plot, file_name):
        plot.savefig(self.location + file_name + '.png')
        plot.clf()
        return True
