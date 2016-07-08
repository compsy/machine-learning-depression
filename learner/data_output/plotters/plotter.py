class Plotter:

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def return_file(self, plot, file_name):
        return (plot, file_name)