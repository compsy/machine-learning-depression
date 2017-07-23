import os
from learner.data_output.std_logger import L

class DatatoolOutput:

    @staticmethod
    def export(variablename, value):
        value = str(value)
        variablename = str(variablename)
        with open("exports/z_variables.dat", "a+") as myfile:
            myfile.write(variablename + " = " + value + "\n")

    @staticmethod
    def clear():
        try:
            os.remove("exports/z_variables.dat")
        except OSError:
            pass
