import numpy as np
import tensorflow as tf
from tensorflow import keras as K


def anneal_rate(epoch, minv=0.1, maxv=5.0):
    import math
    return math.log(maxv/minv) / epoch


class ScheduledVariable:
    """General variable which is changed during the course of training according to some schedule"""

    def __init__(self, name="variable",):
        self.variable = tf.Variable(self.value(0), name=name)

    def value(self, epoch):
        """Should return a scalar value based on the current epoch.
Each subclasses should implement a method for it."""
        pass

    def update(self, epoch, logs):
        self.variable.assign(self.value(epoch))


class GumbelSoftmax(ScheduledVariable):
    count = 0

    def __init__(self, N, M, min_temp, max_temp, full_epoch, annealer=anneal_rate,
                 beta=1., offset=0, train_gumbel=True, train_softmax=True,
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

    def call(self, logits):
        u = tf.random.uniform(tf.shape(logits), 0, 1)
        gumbel = tf.math.subtract(
            0, tf.math.log(
                tf.math.subtract(1e-20, tf.math.log(u + 1e-20))
            )
        )

        if self.train_gumbel:
            train_logit = logits + gumbel
        else:
            train_logit = logits

        if self.test_gumbel:
            test_logit = logits + gumbel
        else:
            test_logit = logits

        def softmax_train(x):
            return K.activations.softmax(x / self.variable)

        def argmax_train(x):
            # use straight-through estimator
            argmax = tf.one_hot(tf.math.argmax(x),
                                self.M, on_value=1, off_value=0)
            softmax = K.activations.softmax(x / self.variable)
            return tf.stop_gradient(argmax-softmax) + softmax

        def softmax_test(x):
            return K.activations.softmax(x / self.min)

        def argmax_test(x):
            return tf.one_hot(tf.math.argmax(x),
                              self.M, on_value=1, off_value=0)

        if self.train_softmax:
            train_activation = softmax_train
        else:
            train_activation = argmax_train

        if self.test_softmax:
            test_activation = softmax_test
        else:
            test_activation = argmax_test

        return K.backend.in_train_phase(
            train_activation(train_logit),
            test_activation(test_logit))

    def __call__(self, previous_layer):
        GumbelSoftmax.count += 1
        c = GumbelSoftmax.count-1

        layer = K.layers.Lambda(self.call, name="gumbel_{}".format(c))

        logits = K.layers.Reshape((self.N, self.M))(previous_layer)
        q = K.activations.softmax(logits)
        log_q = tf.math.log(q + 1e-20)
        loss = tf.math.reduce_mean(q * log_q) * self.beta

        layer.add_loss(K.backend.in_train_phase(loss, 0.0), logits)

        return layer(logits)

    def value(self, epoch):
        return np.max([
            self.min,
            self.max * np.exp(- self.anneal_rate * max(epoch - self.offset, 0))
        ])
