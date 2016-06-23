import numpy as np
import collections


class SingleOutputFrameCreator:

    def create_single_frame(self, questionnaires, participants):
        dataFrame = collections.namedtuple('DataFrame', ['data', 'header'])
        rows = len(participants)

        header = ['pident']

        # The number of columns is 1 since we add the pident
        self.cols = len(header)

        # Determine the number of columns in the eventual dataframe
        for questionnaire in questionnaires:
            self.cols += questionnaire.number_of_variables()
            header.extend(questionnaire.get_header())

        header = np.array(header)
        result = np.empty(rows * self.cols).reshape(rows, self.cols)

        # TODO: This is quite slow and inefficient, should be refactored
        for index, participant_key in enumerate(participants):
            print('Processing participant: ', participant_key)
            participant = participants[participant_key]

            # initialize an empty array
            participant_array = [None] * self.cols

            # First add the pident to the array
            participant_array[0] = participant.pident

            last_index = 1
            for col_index, questionnaire in enumerate(questionnaires):
                row = questionnaire.get_data_export(participant)
                for value in row:
                    participant_array[last_index] = value
                    last_index += 1

            print(participant_array)
            result[index] = participant_array

        return dataFrame(data=result, header=header)
