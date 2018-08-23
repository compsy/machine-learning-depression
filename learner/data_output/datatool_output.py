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

    @staticmethod
    def number_to_string(number):
        if number == 0:
            return 'zero'
        if number == 1:
            return 'one'
        if number == 2:
            return 'two'
        if number == 3:
            return 'three'
        if number == 4:
            return 'four'
        if number == 5:
            return 'five'
        if number == 6:
            return 'six'
        if number == 7:
            return 'seven'
        if number == 8:
            return 'eight'
        if number == 9:
            return 'nine'
        if number == 10:
            return 'ten'
