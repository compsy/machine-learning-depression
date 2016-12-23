import inspect
import numpy as np

from sklearn.decomposition import PCA


class Questionnaire:

    def __init__(self, name, filename, measurement_moment, reader, function_mapping, other_available_variables=[]):
        self.name = name
        self.filename = filename
        self.measurement_moment = measurement_moment
        self.data = self.create_data_hash(reader.read_file(filename))

        # Create the function mapping
        raw_value_function_mapping = self.create_raw_value_function_mapping(other_available_variables)

        # Merge the two functions into one
        self.function_mapping = function_mapping.copy()
        self.function_mapping.update(raw_value_function_mapping)

        #pca = PCA(n_components=2)
        #pca.fit(x_data)
        #print(pca.explained_variance_ratio_)

    def create_raw_value_function_mapping(self, other_available_variables):
        """
        Creates the value function mappings for the raw parameters

        Parameters
        ----------
        other_available_variables - all variables available in the file

        Returns
        -------
        dict with the mapping. Note that variables will be stored as a string. This will be worked around in the method
        accessing the data.

        """
        function_dict = {}
        for variable_name in other_available_variables:
            function_dict[variable_name] = variable_name
        return function_dict

    def create_data_hash(self, data):
        data_hashed = {}

        key = 'pident'
        for index, entry in data.iterrows():
            # Only use lowercase keys
            temp = entry.rename(lambda keyname: keyname.lower())
            data_hashed[int(temp[key])] = temp

        return data_hashed

    def get_header(self):
        col_names = self.function_mapping.keys()
        return map(lambda name: self.variable_name(self.name + '-' + name, force_lower_case=False), col_names)

    def variable_name(self, variable, force_lower_case):
        measurement_moment = self.measurement_moment
        if force_lower_case:
            measurement_moment = measurement_moment.lower()

        return measurement_moment + variable

    def number_of_variables(self):
        return len(self.function_mapping.keys())

    def get_data_export(self, participant):
        """
        Function that returns all data that is ment to be exported from the class (all things in the function mapping)
        Parameters
        ----------
        participant - the participant for whom the variables should be exported

        Returns
        -------
        array containing the data
        """
        participant_key = participant.pident
        res = [None] * self.number_of_variables()

        all_functions = inspect.getmembers(self, predicate=inspect.ismethod)
        all_functions = list(map(lambda function: function[1], all_functions))

        if participant_key in self.data:
            for index, field in enumerate(self.function_mapping.keys()):
                possible_function = self.function_mapping[field]

                # If the passed value is a function, call it, otherwise it should be a raw value
                if (possible_function in all_functions):
                    res[index] = possible_function(participant)
                else:
                    res[index] = self.get_field(participant, possible_function)

        return res

    def get_field(self, participant, field, force_lower_case=False):
        dat = self.get_row(participant)
        q_name = self.variable_name(field, force_lower_case=force_lower_case)
        if q_name in dat:
            val = dat[q_name]

            # NOTE Return NaN if we still find a negative value here
            return val if val is not None and val >= 0 else np.nan
        return None

    def get_row(self, participant):
        key = participant.pident
        dat = []
        if key in self.data:
            dat = self.data[participant.pident]

        return dat

    ### Abstract methods
    def som_score(self, participant):
        raise NotImplementedError
