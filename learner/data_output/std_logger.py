import logging
import warnings

class L:
    @staticmethod
    def setup():
        FORMAT = '%(asctime)-15s -> %(message)s'
        logging.basicConfig(filename='../exports/output.log', format=FORMAT, level=logging.INFO)
        for i in range(10): L.br()
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