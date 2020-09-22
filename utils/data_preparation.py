import tensorflow as tf


class Normalizer:
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def normalize(self, x):
        return tf.math.divide(
            tf.subtract(x, self.xmin),
            (self.xmax - self.xmin)
        )

    def denormalize(self, x):
        return tf.add(
            tf.multiply((self.xmax - self.xmin), x),
            self.xmin
        )
