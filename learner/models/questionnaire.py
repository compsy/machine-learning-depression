class Questionnaire:

    def __init__(self, name, filename, measurement_moment, reader, function_mapping):
        self.name = name
        self.filename = filename
        self.measurement_moment = measurement_moment
        self.data = self.createDataHash(reader.read_file(filename))
        self.function_mapping = function_mapping

    def createDataHash(self, data):
        data_hashed = {}

        # Some files have a capitalized version of Pident.
        key = 'pident'
        if key not in data:
            key = 'Pident'

        for index, entry in data.iterrows():
            data_hashed[int(entry[key])] = entry
        return data_hashed

    def get_header(self):
        col_names = self.function_mapping.keys()
        return map(lambda name: self.variable_name(self.name + '-' + name), col_names)

    def variable_name(self, variable):
        return self.measurement_moment + variable

    def number_of_variables(self):
        return len(self.function_mapping.keys())

    def get_row(self, participant):
        key = participant.pident
        dat = []
        if key in self.data:
            dat = self.data[participant.pident]

        return dat

    def get_data(self, participant):
        key = participant.pident
        res = [None] * self.number_of_variables()

        if key in self.data:
            for index, field in enumerate(self.function_mapping.keys()):
                res[index] = self.function_mapping[field](participant)

        return res

    def get_field(self, participant, field):
        dat = self.get_row(participant)
        q_name = self.variable_name(field)
        print(q_name)
        if q_name in dat:
            return dat[q_name]
        return None

    ### Abstract methods
    def som_score(self, participant):
        raise NotImplementedError
