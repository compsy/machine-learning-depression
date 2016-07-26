import numpy as np
from data_output.std_logger import L


class CsvExporter:

    @staticmethod
    def export(filename, data, header):
        string_header = ','.join(header)
        L.info('Exporting data to: ' + filename)
        np.savetxt(filename, data, delimiter=',', header=string_header, fmt='%s')
