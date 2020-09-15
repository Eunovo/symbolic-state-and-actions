import tensorflow as tf
from tensorflow.Keras import layers, Sequential

class OptionSelector:
    def __init__(self, n_state_bits):
        self.model = Sequential([
            layers.Input(shape=(n_state_bits,)),
            layers.Dense((2 * n_state_bits), activation='relu'),
            layers.Dense(n_state_bits, activation='softmax')
        ]);

    def select_option(self, state):
        return tf.argmax(
            self.model.predict(state)
        )
    
    def receive_reward(self, reward, tape, optimizer):
        pass