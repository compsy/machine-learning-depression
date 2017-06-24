import os
import pickle
from learner.caching.cacher import Cacher


class ObjectCacher():

    def __init__(self, directory='cache/'):
        self.directory = directory

        # Create the directory if it does not yet exist
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_dirred_file(self, cache_name):
        return self.directory + cache_name

    def read_cache(self, cache_name):
        cache_name = self.get_dirred_file(cache_name)
        if not self.file_available(cache_name, add_dir=False):
            raise FileNotFoundError('File: ' + cache_name + ' not found!')

        with open(cache_name, 'rb') as input:
            data = pickle.load(input)
            return data

    def write_cache(self, data, cache_name):
        cache_name = self.get_dirred_file(cache_name)
        with open(cache_name, 'wb') as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

    def file_available(self, cache_name, add_dir=True):
        if add_dir: cache_name = self.get_dirred_file(cache_name)
        return os.path.isfile(cache_name)

    def files_in_dir(self):
        return os.listdir(self.directory)
