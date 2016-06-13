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

    def getHeader(self):
        col_names = self.function_mapping.keys()
        return map(lambda name: self.variableName(self.name+'-'+name), col_names)

    def variableName(self, variable):
        return self.measurement_moment + variable

    def numberOfVariables(self):
        return len(self.function_mapping.keys())

    def getRow(self, participant):
        key = participant.pident
        dat = []
        if key in self.data:
            dat = self.data[participant.pident]

        return dat

    def getData(self, participant):
        key = participant.pident
        res = [None] * self.numberOfVariables()

        if key in self.data:
            for index, field in enumerate(self.function_mapping.keys()):
                res[index] = self.function_mapping[field](participant)

        return res

    def getField(self, participant, field):
        dat = self.getRow(participant)
        q_name = self.variableName(field)
        if q_name in dat:
            return dat[q_name]
        return None

    ### Abstract methods
    def somScore(self, participant):
        raise 'Method should be implemented by subclass'
