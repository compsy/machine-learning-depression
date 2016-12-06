import numpy as np
import pandas
import inspect
from learner.data_output.csv_exporter import CsvExporter
from rpy2.robjects import pandas2ri
import pytest


class TestCsvExporter():
    def test_csvexporter_exposes_a_static_function(self):
        static_methods = inspect.getmembers(CsvExporter, predicate=inspect.isfunction)
        assert static_methods[0][0] == 'export'

    def test_export_filename(self):
        filename = '../tests/data_examples/test.csv'
        data = [[1,2,3,4],[5,6,7,8]]
        header = ['c','a','b','c']
        CsvExporter.export(filename, data, header)

        # Get the results
        my_file = open(filename)
        result = my_file.read()
        expected = '# c,a,b,c\n1,2,3,4\n5,6,7,8\n'
        assert result == expected

