import logging
import warnings
import time


class L:

    @staticmethod
    def setup():
        date = time.strftime("%y%m%d-%H%M")
        FORMAT = '%(asctime)-15s -> %(message)s'
        logging.basicConfig(filename='../exports/' + date + '_output.log', format=FORMAT, level=logging.INFO)
        L.info('Starting Machine Learning')

    @staticmethod
    def info(text):
        print(text)
        logging.info(text)

    @staticmethod
    def br():
        print('')
        logging.info('')

    @staticmethod
    def debug(text):
        print(text)
        logging.debug(text)

    @staticmethod
    def warn(text):
        warnings.warn(text)
        logging.debug(text)
