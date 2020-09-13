import numpy as np
import tensorflow as tf
from tensorflow import keras as K


class GumbelSoftmaxLayer(K.layers.Layer):
    def __init__(self, N, M, min_temp, max_temp, full_epoch, beta=1.):
        super(GumbelSoftmaxLayer, self).__init__()
        self.reshape = K.layers.Reshape((N, M))
        self.beta = beta
        self.gumbel_activation = GumbelSoftmax(
            N, M, min_temp, max_temp, full_epoch, beta=beta)

    def call(self, inputs):
        x = self.reshape(inputs)

        # a = K.activations.softmax(x)
        # log_a = tf.math.log(a + 1e-20)
        # loss = tf.math.reduce_mean(a * log_a) * self.beta
        # self.add_loss(K.backend.in_train_phase(loss, 0.0))

        return self.gumbel_activation(x)

    def get_update_callback(self):
        return K.callbacks.LambdaCallback(
            on_epoch_end=self.gumbel_activation.update
        )


def anneal_rate(epoch, minv=0.1, maxv=5.0):
    import math
    return math.log(maxv/minv) / epoch


class ScheduledVariable:
    """General variable which is changed during the course of training according to some schedule"""

    def __init__(self, name="variable",):
        self.variable = tf.Variable(self.value(0), name=name, dtype=tf.float32)

    def value(self, epoch):
        """Should return a scalar value based on the current epoch.
Each subclasses should implement a method for it."""
        pass

    def update(self, epoch, logs):
        self.variable.assign(self.value(epoch))


class GumbelSoftmax(ScheduledVariable):
    def __init__(self, N, M, min_temp, max_temp, full_epoch, annealer=anneal_rate,
                 beta=1., offset=0, train_gumbel=True, train_softmax=False,
                 test_gumbel=False, test_softmax=False,
                 ):
        self.N = N
        self.M = M
        self.min = min_temp
        self.max = max_temp
        self.train_gumbel = train_gumbel
        self.train_softmax = train_softmax
        self.test_gumbel = test_gumbel
        self.test_softmax = test_softmax
        self.anneal_rate = annealer(
            full_epoch-offset, minv=min_temp, maxv=max_temp)
        self.offset = offset
        self.beta = beta
        super(GumbelSoftmax, self).__init__("temperature")

    def get_gumbel_logits(self, logits):
        u = tf.random.uniform(tf.shape(logits), 0, 1)
        gumbel = tf.negative(tf.math.log(
            tf.math.subtract(1e-20, tf.math.log(u + 1e-20))
        ), name='gumbel')

        if K.backend.in_train_phase(self.train_gumbel, self.test_gumbel):
            return tf.add(logits, gumbel)
        else:
            return logits

    def one_hot(self, logits):
        # print('Logits', logits)
        # print()
        # print('Argmax', tf.math.argmax(logits, axis=2))
        # print()
        return tf.one_hot(tf.math.argmax(logits, axis=2),
                          self.M, on_value=1, off_value=0,
                          dtype=tf.float32)

    def train_activation(self, logits):
        softmax = K.activations.softmax(logits / self.variable)
        if self.train_softmax:
            return softmax
        else:
            # use straight-through estimator
            argmax = self.one_hot(logits)
            stop_gradient = tf.stop_gradient(tf.subtract(argmax, softmax))
            # softmax is only used in back propagation
            return tf.add(stop_gradient, softmax)

    def test_activation(self, logits):
        if self.test_softmax:
            return K.activations.softmax(logits / self.min)
        else:
            one_hot = self.one_hot(logits)
            # print(one_hot)
            # print()
            return one_hot

    def __call__(self, logits):
        logits = self.get_gumbel_logits(logits)
        return K.backend.in_train_phase(
            self.train_activation(logits),
            self.test_activation(logits)
        )

    def value(self, epoch):
        return np.max([
            self.min,
            self.max * np.exp(- self.anneal_rate * max(epoch - self.offset, 0))
        ])
