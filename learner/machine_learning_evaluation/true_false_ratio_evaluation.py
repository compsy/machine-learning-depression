from learner.machine_learning_evaluation.evaluation import Evaluation


class TrueFalseRationEvaluation(Evaluation):

    def __init__(self, pos_label=0):
        super().__init__(name='True False evaluation', model_type='classification')
        self.pos_label = pos_label

    def evaluate(self, y_train, y_test):
        """
        Generates the true / false ratio of the dataset (how many instances are true, how many are false, etc).
        :param y_train: the y values in the training set
        :param y_test: the y values in the test set
        :return: a tuple with: the total number of true instances in the train set, the percentage true in the train
                               set, the total true instances in the test set and the percentage true in the test set
        """
        trues = {'y_train': 0, 'y_test': 0}
        falses = {'y_train': 0, 'y_test': 0}
        for i in y_train:
            if i == self.pos_label:
                trues['y_train'] += 1
            else:
                falses['y_train'] += 1
        for i in y_test:
            if i == self.pos_label:
                trues['y_test'] += 1
            else:
                falses['y_test'] += 1

        train_outcome = (trues['y_train'] / (trues['y_train'] + falses['y_train'])) * 100
        test_outcome = (trues['y_test'] / (trues['y_test'] + falses['y_test'])) * 100
        return (trues['y_train'], train_outcome, trues['y_test'], test_outcome)
