from data_output.std_logger import L


class Evaluation:

    def __init__(self, name='Evaluator', model_type='models'):
        self.model_type = model_type
        self.name = name

    def print_evaluation(self, model, y_true, y_pred):
        evaluation = self.evaluate(y_true, y_pred)
        L.info("\t -> %s for %s: %0.4f" % (self.name, model.given_name, evaluation))

    @property
    def problem_type(self):
        return self.model_type
