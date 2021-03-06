import logging
import warnings
import time


class L:

    @staticmethod
    def setup(logger_hpc):
        global logger_on_hpc
        logger_on_hpc = logger_hpc

        if not logger_hpc:
            date = time.strftime("%y%m%d-%H%M")
            FORMAT = '%(asctime)-15s -> %(message)s'
            logging.basicConfig(filename='exports/' + date + '_output.log', format=FORMAT, level=logging.INFO)
            L.info('Starting Machine Learning')
        print(logger_on_hpc)

    @staticmethod
    def info(text, force=False):
        if not 'logger_on_hpc' in globals():
            L.warn('LOGGER - SETUP SHOULD BE CALLED FIRST')
            L.setup(False)

        if logger_on_hpc and not force:
            rank = MPI.COMM_WORLD.Get_rank()
            if rank != 0:
                return

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
