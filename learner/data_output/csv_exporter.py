import pandas as pd
from learner.data_output.std_logger import L


class CsvExporter:

    @staticmethod
    def export(filename, data):
        L.info('Exporting data to: ' + filename)
        data.to_csv(path_or_buf=filename, sep=',')
