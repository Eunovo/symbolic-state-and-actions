import tensorflow as tf

from layers.gumbel_softmax import GumbelSoftmaxLayer


class StateAutoEncoder:
    def __init__(self, n_epochs, steps_per_epoch, batch_size):
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size

        self.gumbel_layer = GumbelSoftmaxLayer(10, 2, 5.0, 0.7, n_epochs)

        self.encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(1,), batch_size=batch_size),
            tf.keras.layers.Dense(40, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(20),
            self.gumbel_layer,
            tf.keras.layers.Flatten()
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(20,), batch_size=batch_size),
            tf.keras.layers.Dense(100, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(50, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        self.state_autoencoder = tf.keras.Sequential([
            tf.keras.Input(shape=(1,), batch_size=batch_size),
            self.encoder,
            self.decoder,
        ])

    def use_checkpoint(self, checkpoint_dir):
        self.state_autoencoder.save_weights(checkpoint_dir)
        load_status = self.state_autoencoder.load_weights(checkpoint_dir)
        # assert that all model variables have been restored
        load_status.assert_consumed()

    def compile(self):
        self.state_autoencoder.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['accuracy']
        )

    def fit(self, dataset, callbacks=[]):
        callbacks.append(self.gumbel_layer.get_update_callback())

        return self.state_autoencoder.fit(
            dataset, epochs=self.n_epochs,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=callbacks
        )
