import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import categorical_projection_network
from tf_agents.networks import encoding_network, utils
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils

from layers import GumbelSoftmaxLayer
from models.option_actor_network import OptionActorNetwork


class HierachyActorNetwork(network.DistributionNetwork):
    def __init__(
        self,
        observation_spec,
        action_spec,
        n_iterations,
        n_options=4,
        preprocessing_layers=None,
        preprocessing_combiner=None,
        name='HierachyActorNetwork'
    ):
        self.n_options = n_options
        self.action_spec = action_spec

        self.projection_nets = tf.nest.map_structure(
            lambda spec: categorical_projection_network
            .CategoricalProjectionNetwork(spec),
            self.action_spec
        )
        self._output_spec = tf.nest.map_structure(
            lambda proj_net: proj_net.output_spec, self.projection_nets)

        super(HierachyActorNetwork, self).__init__(
            input_tensor_spec=observation_spec,
            output_spec=self._output_spec,
            state_spec=(), name=name
        )

        if (n_options < 1):
            raise ValueError(
                "Number of options must be at least {}".format(n_options)
            )

        fc_layer_params = (100,)

        def create_option():
            return tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(fc_layer_params)),
                tf.keras.layers.Dense(100, activation='relu'),
                tf.keras.layers.Dense(50, activation='relu')
            ])

        self.options = [create_option() for n in range(n_options)]

        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1. / 3.,
            mode='fan_in',
            distribution='uniform'
        )
        self.encoder = encoding_network.EncodingNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=kernel_initializer,
            batch_squash=False
        )
        self.hidden_selector_layer = tf.keras.layers.Dense(
            self.n_options, activation='relu')
        self.selector = GumbelSoftmaxLayer(
            self.n_options, 1, 5.0, 0.7, n_iterations)
        # Remember to update its temperature using it's callback

    @property
    def output_tensor_spec(self):
        return self.action_spec

    def call(
        self,
        observations,
        step_type=(),
        network_state=(),
        training=False,
        mask=None
    ):
        outer_rank = nest_utils.get_outer_rank(
            observations, self.input_tensor_spec)
        # We use batch_squash here in case the observations have a time sequence
        # compoment.
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(
            batch_squash.flatten, observations)

        # we ignore next_state from the encoder
        state, network_state = self.encoder(
            observations,
            step_type=step_type,
            network_state=network_state
        )
        l_hidden = self.hidden_selector_layer(state)
        selection_vector = tf.transpose(
            self.selector(l_hidden), perm=[0, 2, 1]
        )

        options = [
            option(
                state
            ) for option in self.options
        ]
        options = tf.stack(options)
        options = tf.transpose(
            options, perm=[1, 0, 2])

        # select an option using the selection_vector from the master network
        state = tf.matmul(selection_vector, options)

        state = batch_squash.unflatten(state)

        def call_projection_net(proj_net):
            distribution, _ = proj_net(
                state, outer_rank, training=training, mask=mask)
            return distribution

        output_actions = tf.nest.map_structure(
            call_projection_net, self.projection_nets)

        return output_actions, network_state

    def get_options(self):
        return [
            OptionActorNetwork(
                self.input_tensor_spec,
                self._output_spec,
                self.action_spec,
                self.encoder,
                option_model,
                self.projection_nets
            ) for option_model in self.options
        ]
