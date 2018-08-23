from tabulate import tabulate
import numpy as np
from learner.data_output.std_logger import L


class LatexTableExporter:

    @staticmethod
    def export(filename, data, header):
        L.info('Exporting LaTex data to: ' + filename)
        f = open(filename, 'w')
        f.write(tabulate(data, header, tablefmt="latex_booktabs"))
        f.close()
