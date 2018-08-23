from .data_transformer import DataTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

class OneHotEncoderTransformer(DataTransformer):
    @staticmethod
    def perform_encoding(data, limit, split_binary=False):

        # First we fit the encoder to determine the number of options in each variable
        enc = OneHotEncoder()
        enc.fit(data)
        if split_binary:
            correct_column_locations = enc.n_values_ < limit
        else:
            correct_column_locations = np.logical_and(enc.n_values_ < limit, enc.n_values_ > 2)

        updated_indices = np.where(correct_column_locations)[0]
        normal_indices = np.where(np.invert(correct_column_locations))[0]

        old_names = data.columns[normal_indices]
        updated_names = data.columns[updated_indices]

        # Then we refit it only on the updated_indices that are considered categorical.
        enc = OneHotEncoder(categorical_features=updated_indices, sparse=False)
        enc.fit(data)
        new_data = enc.transform(data)

        new_names = []

        for i in range(len(updated_indices)):
            start = enc.feature_indices_[i]
            end = enc.feature_indices_[i + 1]

            current_indices = range(start, end)
            used_indices = set(current_indices).intersection(enc.active_features_)
            current_column = updated_names[i]
            for j in range(0, len(used_indices)):
                new_names.append(current_column + str(j + 1))

        # The old features are added to the right of the matrix
        new_names.extend(old_names)

        assert len(new_names) == np.shape(new_data)[1]
        new_data = pd.DataFrame(new_data, columns=new_names)
        return new_data

        # 262

        # THIS IS NOT POSSIBLE! We have to use the indices in feature indices etc.
        # for i in range(len(updated_names)):

    # current = n_values[i]
    # print(current)
    # for j in range(0, current):
    # print('>' + str(j))
    # new_names.append(updated_names[i] + str(j))
    # print(len(new_names) + len(old_names) - np.shape(new_data)[1])
    # for i in len(enc.feature_indices_) - 1:
    # start = feature_indices[i]
    # end   = feature_indices[i+1]

    # current_indices = range(start,end)

    # # Remask the set of updated_indices here.
    # used_indices = set(current_indices).intersection(active_features)
    # current_column = data.columns[min(used_indices)]

    # while all_columns_index < min(used_indices):
    # new_names.append(data.columns[all_columns_index])
    # all_columns_index += 1

    # for j in used_indices:
    # new_names.append(current_column + str(j))
