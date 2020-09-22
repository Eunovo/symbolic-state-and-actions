import tensorflow as tf
import os

from layers import GumbelSoftmaxLayer
from utils import Normalizer


class StateAutoEncoder:
    def __init__(
        self,
        n_epochs,
        steps_per_epoch,
        n_encode_bits,
        normalize=False,
        normalizer=Normalizer,
        min_value=None,
        max_value=None
    ):
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.n_encode_bits = n_encode_bits
        self.normalize = normalize
        self.normalizer = Normalizer(min_value, max_value)

        if (self.normalize and not min_value and not max_value):
            raise ValueError(
                "min_value and max_value are required for normalization")

        self.callbacks = []

        gumbel_layer = GumbelSoftmaxLayer(
            self.n_encode_bits, 2, 5.0, 0.7, n_epochs)
        self.callbacks.append(gumbel_layer.get_update_callback())

        self.encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(40, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(20),
            gumbel_layer,
            tf.keras.layers.Flatten()
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(20,)),
            tf.keras.layers.Dense(40, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(20, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        self.state_autoencoder = tf.keras.Sequential([
            tf.keras.Input(shape=(1,)),
            self.encoder,
            self.decoder,
        ])

    def use_checkpoints(self, checkpoint_dir):
        checkpoint_file = checkpoint_dir + "ckpt"
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_file,
                save_weights_only=True
            )
        )
        if (len(os.listdir(checkpoint_dir)) > 0):
            print("Restoring from", checkpoint_file)
            self.state_autoencoder.load_weights(checkpoint_file)

    def compile(self):
        self.state_autoencoder.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['accuracy']
        )

    def fit(self, dataset, callbacks=[]):
        self.callbacks.extend(callbacks)

        if (self.normalize):
            def normalize(x):
                normalized_x = self.normalizer.normalize(x)
                return ([normalized_x], [normalized_x])

            dataset = dataset.map(normalize)

        return self.state_autoencoder.fit(
            dataset, epochs=self.n_epochs,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=self.callbacks
        )

    def encode(self, data):
        if (self.normalize):
            data = self.normalizer.normalize(data)
        return self.encoder.predict(data)

    def decode(self, data):
        result = self.decoder.predict(data)

        if (self.normalize):
            result = self.normalizer.denormalize(data)

        return result

    def save(self, save_dir):
        self.state_autoencoder.save(save_dir)

    def load_model(self, save_dir):
        self.state_autoencoder = tf.keras.models.load_model(save_dir)
