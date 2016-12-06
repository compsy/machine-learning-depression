import numpy as np
import pandas
import inspect
from learner.data_output.latex_table_exporter import LatexTableExporter
from rpy2.robjects import pandas2ri
import pytest


class TestLatexTableExporter():
    def test_latex_table_exporter_exposes_a_static_function(self):
        static_methods = inspect.getmembers(LatexTableExporter, predicate=inspect.isfunction)
        assert static_methods[0][0] == 'export'

    def test_export_filename(self):
        filename = '../tests/data_examples/test.tex'
        data = [[1,2,3,4],[5,6,7,8]]
        header = ['c','a','b','c']
        LatexTableExporter.export(filename, data, header)

        # Get the results
        my_file = open(filename)
        result = my_file.read()
        expected = '\\begin{tabular}{rrrr}\n'\
        '\hline\n'\
        '   c &   a &   b &   c \\\\\n'\
        '\hline\n'\
        '   1 &   2 &   3 &   4 \\\\\n'\
        '   5 &   6 &   7 &   8 \\\\\n'\
        '\hline\n'\
        '\end{tabular}'
        assert result == expected

