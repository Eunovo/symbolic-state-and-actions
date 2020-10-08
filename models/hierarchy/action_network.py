import tensorflow as tf
from tensorflow import keras as K

from layers import GumbelSoftmaxLayer


class ActionNetwork(K.Model):
    def __init__(self, num_actions, num_state_bits, n_epochs):
        super(ActionNetwork, self).__init__()
        self.inputs = K.layers.Dense((num_actions + num_state_bits), activation='relu')
        self.hidden_1 = K.layers.Dense(1000, activation='relu')
        self.hidden_2 = K.layers.Dense(num_state_bits * 2)
        self.gumbel = GumbelSoftmaxLayer(num_state_bits, 2, 0.7, 5.0, n_epochs)
        self.outputs = K.layers.Lambda(lambda x: x[:, :, 0])

    @property   
    def gumbel_callback(self):
        return self.gumbel.get_update_callback()

    def call(self, inputs):
        x = self.inputs(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.gumbel(x)
        
        return self.outputs(x)
