import numpy as np
import pandas as pd
import collections


class SingleOutputFrameCreator:

    def create_single_frame(self, questionnaires, participants):
        dataFrame = collections.namedtuple('DataFrame', ['data', 'header'])
        rows = len(participants)

        header = self.create_header(questionnaires, add_pident=True)
        cols = len(header)

        result = np.empty(rows * cols).reshape(rows, cols)

        # TODO: This is quite slow and inefficient, should be refactored
        for index, participant_key in enumerate(participants):
            # print('Processing participant: ', participant_key)
            participant = participants[participant_key]

            # initialize an empty array
            participant_array = [None] * cols

            # First add the pident to the array
            participant_array[0] = participant.pident

            last_index = 1
            for col_index, questionnaire in enumerate(questionnaires):
                row = questionnaire.get_data_export(participant)
                for value in row:
                    participant_array[last_index] = value
                    last_index += 1

            # print(participant_array)
            result[index] = participant_array

        return pd.DataFrame(result, columns=header)

    def create_header(self, questionnaires, add_pident=True):
        """docstring for create_header"""
        header = []
        if add_pident: header = ['pident']

        # Determine the number of columns in the eventual dataframe
        for questionnaire in questionnaires:
            header.extend(questionnaire.get_header())

        header = np.array(header)
        return (header)
