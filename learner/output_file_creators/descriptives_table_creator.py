from learner.data_output.latex_table_exporter import LatexTableExporter
import numpy as np

from learner.data_output.std_logger import L


class DescriptivesTableCreator():

    @staticmethod
    def generate_coefficient_descriptives_table(x_data, x_names, coefficients, name):
        """
        Creates a latex table of the coefficients as provided in the function. Usually these coefficients are generated
        using elastic net of some sort.
        :param x_data: the original x_data (needed for calculating some descriptives)
        :param x_names: the names of the x variables
        :param coefficients: the coefficients for each x_name
        :param name: the name of the file to export
        """
        header = []
        header.append('#')
        header.append('Feature')
        header.append('SD')
        header.append('Mean')

        ranks = list(range(1, 26))
        types = []
        for column in x_data.T:
            unique_items = len(np.unique(column))
            if unique_items <= 2:
                types.append('Dichotomous')
            elif unique_items <= 10:
                types.append('Categorical')
            else:
                types.append('Discrete')

        if coefficients is not None:
            header.append('Elastic net Coefficient')
            data = list(zip(ranks, x_names, x_data.std(0), x_data.mean(0), coefficients[0:, 1], types))
        else:
            data = list(zip(ranks, x_names, x_data.std(0), x_data.mean(0), types))
        header.append('Type')
        LatexTableExporter.export('exports/' + name + '.tex', data, header)

    @staticmethod
    def create_data_descriptive_plots(participants, x_data, x_names):
        ages = []
        genders = []
        for index, participant_key in enumerate(participants):
            participant = participants[participant_key]
            genders.append(participant.gender)
            ages.append(participant.age)

        gender_output = np.bincount(genders)
        if len(gender_output) is not 2:
            L.warn('There are more than 2 types of people in the DB')
            L.warn(genders)

        gender_output = ((gender_output[0] / len(participants)) * 100, (gender_output[1] / len(participants)) * 100)
        ages_output = (len(participants), np.average(ages), np.median(ages), np.std(ages), min(ages), max(ages))

        L.info('The participants (%d) have an average age of %0.2f, median %0.2f, sd %0.2f, range %d-%d' % ages_output)
        L.info('The participants are %0.2f percent male (%0.2f percent female)' % gender_output)
        self.data_density_plotter.plot(x_data, x_names)
