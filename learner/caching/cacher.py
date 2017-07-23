from learner.data_output.std_logger import L
import os


class Cacher:
    """
    Super class for all cahcers
    """

    @staticmethod
    def is_valid_cache(cache, keys):
        """
        Function to determine whether the loaded cache is valid
        """
        valid_cache = all(key in cache for key in keys)
        if not valid_cache:
            L.warn('Skipping model because it is corrupt..')
        return valid_cache

    @staticmethod
    def clean_cache(thedir='cache/mlmodels/'):
        thedir = thedir if thedir.endswith('/') else thedir + '/'
        filelist = [ f for f in os.listdir(thedir) if f.endswith(".pkl") ]
        for f in filelist:
            os.remove(thedir + f)

