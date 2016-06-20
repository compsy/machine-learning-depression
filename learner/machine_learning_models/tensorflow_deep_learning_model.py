from .machine_learning_model import MachineLearningModel

import tensorflow as tf
from subprocess import call
import numpy as np

class TensorFlowDeepLearningModel(MachineLearningModel):

    def train(self):
        if (self.skmodel is not None):
            return self

        # The logdir is used for running tensorboard.
        logdir = '/tmp/tensorlog'

        # number of rows (2981)
        training_examples = self.x_train.shape[0]

        # number of columns
        features = self.x_train.shape[1]

        # neurons = 1

        # Number of features to predict
        output_neurons = 1 # self.y_train.shape[1]

        # Input placeholder
        x = tf.placeholder(tf.float32, [training_examples, features])

        # Weights and Bias. The scope created here is just for the tensorboard to show these neurons nicely
        with tf.name_scope('hidden') as scope:
            W = tf.Variable(tf.zeros([features, output_neurons ]), name='Weights')
            b = tf.Variable(tf.zeros([output_neurons]), name='Bias')

        # Try to find values for W and b that compute y_data = W * x_data + b
        y = tf.nn.softmax(tf.matmul(x, W) + b)

        # Implement cross entropy and mean squared error
        #y_ = tf.placeholder(tf.float32, [None, 10])
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        mse = tf.reduce_mean(tf.square(y - self.y_train))

        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(mse)

        # Before starting, initialize the variables.  We will 'run' this first.
        init = tf.initialize_all_variables()

        # Launch the graph.
        sess = tf.Session()
        sess.run(init)

        tf.train.SummaryWriter(logdir , sess.graph)
        # Fit the line.
        for step in range(1000):
            #batch_xs, batch_ys = x_data.next_batch(100)
            sess.run(train, feed_dict={x: self.x_train, y: self.y_train})
            if step % 20 == 0:
                print(step, sess.run(W), sess.run(b))

        print(sess.run(W), sess.run(b))
        print('Go to: http://localhost:6006')
        print(["tensorboard", "--logdir="+logdir])
        self.skmodel = sess
        self.tfmodel = y
        return self

    def predict(self):
        self.train()
        # number of rows (2981)
        training_examples = self.x_test.shape[0]

        # number of columns
        features = self.x_test.shape[1]

        x = tf.placeholder(tf.float32, [training_examples, features])
        self.skmodel.run(self.tfmodel, feed_dict= {x: self.x_test})
