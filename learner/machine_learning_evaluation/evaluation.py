class Evaluation:
    def __init__(self, name='Evaluator', model_type='regression'):
        self.model_type = model_type
        self.name = name

    def print_evaluation(self, model, y_true, y_pred):
        evaluation = self.evaluate(y_true, y_pred)
        print("\t -> %s of %s: %0.2f" % (self.name, model.given_name, evaluation))

    @property
    def problem_type(self):
        return self.model_type
