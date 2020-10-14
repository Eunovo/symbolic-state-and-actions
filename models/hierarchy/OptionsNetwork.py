import tensorflow as tf

from tf_agents.networks import network
from tf_agents.distributions import gumbel_softmax
from tf_agents.networks import encoding_network, utils
from tf_agents.specs import distribution_spec, tensor_spec
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils


class GumbelLayer(gumbel_softmax.GumbelSoftmax):
    def __init__(
        self,
        temperature=None,
        logits=None,
        dtype=None,
        name='Gumbel_'
    ):
        super(GumbelLayer, self).__init__(
            temperature, logits=logits,
            dtype=dtype, name=name
        )

    @property
    def parameters(self):
        params = super().parameters
        params['temperature'] = super().temperature
        params['logits'] = super().logits
        return params

class OptionsNetwork(network.DistributionNetwork):
    def __init__(
        self,
        observation_spec,
        action_spec,
        n_options,
        name='OptionsNetwork'
    ):
        self.n_options = n_options
        self.action_spec = action_spec

        output_shape = action_spec.shape  # .concatenate([2])
        output_spec = self._output_distribution_spec(
            output_shape, action_spec, name
        )

        super(OptionsNetwork, self).__init__(
            input_tensor_spec=observation_spec,
            output_spec=output_spec,
            state_spec=(), name=name
        )

        self._output_shape = output_shape

        self.inputs = tf.keras.layers.Dense(
            observation_spec.shape.num_elements(),
            activation='relu'
        )
        self.hidden_1 = tf.keras.layers.Dense(
            1000, activation='relu'
        )
        self.projection_layer = tf.keras.layers.Dense(
            self._output_shape.num_elements()
        )

    def _output_distribution_spec(self, output_shape, sample_spec, network_name):
        input_param_spec = {
            'temperature': tensor_spec.TensorSpec(
                (), name=network_name + '_temp'),
            'logits':
                tensor_spec.TensorSpec(
                    shape=output_shape,
                    dtype=tf.float32,
                    name=network_name + '_logits'
                )
        }

        return distribution_spec.DistributionSpec(
            GumbelLayer,
            input_param_spec,
            sample_spec=sample_spec,
            dtype=sample_spec.dtype
        )

    def call(
        self, observations,
        step_type=(), network_state=(),
        training=False, mask=None
    ):
        outer_rank = nest_utils.get_outer_rank(
            observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(
            batch_squash.flatten, observations)

        state = self.inputs(observations)
        state = self.hidden_1(state)
        logits = self.projection_layer(state)
        logits = batch_squash.unflatten(logits)

        output = self.output_spec.build_distribution(temperature=0.7, logits=logits), network_state
        return output
