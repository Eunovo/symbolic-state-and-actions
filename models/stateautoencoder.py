import tensorflow as tf
import os

from layers import GumbelSoftmaxLayer


class StateAutoEncoder:
    def __init__(
        self,
        n_epochs,
        steps_per_epoch,
        n_encode_bits,
        normalize=False,
        normalizer=None
    ):
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.n_encode_bits = n_encode_bits
        self.normalize = normalize
        self.normalizer = normalizer

        if (self.normalize and not normalizer):
            raise ValueError(
                "a normalizer is required for normalization")

        self.callbacks = []

        gumbel_layer = GumbelSoftmaxLayer(
            self.n_encode_bits, 2, 5.0, 0.7, n_epochs)
        self.callbacks.append(gumbel_layer.get_update_callback())

        self.encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(500, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1000, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_encode_bits * 2),
            gumbel_layer,
            tf.keras.layers.Lambda(lambda x: x[:, :, 0]),
        ], name='Encoder')
        # self.encoder.summary()

        self.decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(n_encode_bits,)),
            tf.keras.layers.Dense(1000, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(500, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ], name='Decoder')
        # self.decoder.summary()

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
            loss='BCE',
            metrics=['accuracy']
        )

    def fit(self, dataset, callbacks=[]):
        self.callbacks.extend(callbacks)

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
            result = self.normalizer.denormalize(result)

        return result

    def save(self, save_dir):
        self.state_autoencoder.save(save_dir)

    def load_model(self, save_dir):
        self.state_autoencoder = tf.keras.models.load_model(save_dir)
