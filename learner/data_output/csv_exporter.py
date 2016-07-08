import numpy as np


class CsvExporter:

    @staticmethod
    def export(filename, data, header):
        string_header = ','.join(header)
        print('Exporting data to: ', filename)
        np.savetxt(filename, data, delimiter=',', header=string_header, fmt='%s')
