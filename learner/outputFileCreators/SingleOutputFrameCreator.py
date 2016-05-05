import numpy as np
import collections

class SingleOutputFrameCreator:
    def create_single_frame(self, questionnaires, participants):
        dataFrame = collections.namedtuple('DataFrame', ['data', 'header'])
        rows = len(participants)
        self.cols = 0

        header = []

        # Determine the number of columns in the eventual dataframe
        for questionnaire in questionnaires:
            self.cols += questionnaire.numberOfVariables()
            header.extend(questionnaire.getHeader())

        header = np.array(header)
        result = np.empty(rows * self.cols).reshape(rows, self.cols)

        # TODO: This is quite slow and inefficient, should be refactored
        for index, participant_key in enumerate(participants):
            participant = participants[participant_key]
            participant_array = [None] * self.cols
            last_index = 0
            for col_index, questionnaire in enumerate(questionnaires):
                row = questionnaire.getData(participant)
                for value in row:
                    participant_array[last_index] = value
                    last_index += 1

            result[index] = participant_array

        return dataFrame(data=result, header=header)
