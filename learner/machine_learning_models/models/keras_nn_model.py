from keras.optimizers import Adam
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score

from machine_learning_models.machine_learning_model import MachineLearningModel
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
import numpy as np
from pandas import DataFrame
from data_output.std_logger import L


class KerasWrapper(MachineLearningModel):
    # Override the train function, as the keras API returns a history object, not a trained model
    def train(self):
        self.skmodel.fit(X=self.x_train, y=self.y_train)


class KerasNnModel(KerasWrapper):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='regression', **kwargs)

        # Wrap the model in a scikit api
        self.skmodel = KerasRegressor(build_fn=self.baseline_model, nb_epoch=500, batch_size=32, verbose=1)

    #def validate(self):
    # self.skmodel.evaluate(self.x_test, self.y_test, batch_size=32, verbose=1, sample_weight=None)

    def baseline_model(self):
        # Create the model
        keras_model = Sequential()
        keras_model.add(Dense(output_dim=400, input_dim=np.shape(self.x_data)[1]))
        keras_model.add(Dropout(0.4))
        keras_model.add(Activation("relu"))
        keras_model.add(Dense(output_dim=100))
        keras_model.add(Activation("relu"))
        keras_model.add(Dropout(0.4))
        keras_model.add(Dense(output_dim=100))
        keras_model.add(Activation("relu"))
        keras_model.add(Dropout(0.4))
        keras_model.add(Dense(output_dim=1))
        keras_model.add(Activation("relu"))

        # Create the optimizer with a learning rate
        adam = Adam(lr=0.01)

        keras_model.compile(loss='mean_squared_error', optimizer=adam)
        L.info('Printing Neural Network configuration:')
        keras_model.summary()
        L.info(keras_model.to_json())

        return keras_model

    def xx(self):
        # self.skmodel.fit(self.x_train, self.y_train, nb_epoch=2, batch_size=32)
        L.info(np.shape(np.transpose(self.x_test)))
        err = self.skmodel.predict((self.x_test))
        err = np.ravel(err) - self.y_test
        out = DataFrame(data={'err': err, 'out': np.ravel(self.y_test)})
        out.to_csv('../exports/output_keras.csv')
        # pred = self.skmodel.predict(np.reshape(self.x_test[i], (1, len(x_names))))
        # act = self.y_test[1]
        # err.append(np.ravel(pred[0][0] - act))

        # print(err)


class KerasNnClassificationModel(KerasWrapper):

    def __init__(self, x, y, x_names, y_names, verbosity, **kwargs):
        super().__init__(x, y, x_names, y_names, model_type='classification', **kwargs)

        # Wrap the model in a scikit api
        self.skmodel = KerasClassifier(build_fn=self.baseline_model, nb_epoch=5000, batch_size=64, verbose=1)

    def baseline_model(self):
        # Create the model
        keras_model = Sequential()
        keras_model.add(Dense(output_dim=32, input_dim=len(self.x_names), init='lecun_uniform', activation='sigmoid'))
        keras_model.add(Dropout(0.5))
        keras_model.add(Dense(output_dim=32, input_dim=32, init='lecun_uniform', activation='sigmoid'))
        keras_model.add(Dropout(0.5))
        keras_model.add(Dense(output_dim=32, input_dim=32, init='lecun_uniform', activation='sigmoid'))
        keras_model.add(Dropout(0.5))
        keras_model.add(Dense(output_dim=16, input_dim=32, init='lecun_uniform', activation='sigmoid'))
        keras_model.add(Dropout(0.5))
        keras_model.add(Dense(output_dim=1, init='lecun_uniform', activation='sigmoid'))

        # Create the optimizer with a learning rate
        adam = Adam(lr=0.1)

        keras_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        keras_model.summary()
        L.info('Printing Neural Network configuration:')
        L.info(keras_model.to_json())

        return keras_model
