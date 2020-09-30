import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils


class OptionActorNetwork(network.DistributionNetwork):
    def __init__(
        self,
        input_tensor_spec,
        output_spec,
        action_spec,
        encoder,
        option_model,
        projection_nets,
        name='OptionActorNetwork'
    ):
        super(OptionActorNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            output_spec=output_spec,
            state_spec=(), name=name
        )
        self.encoder = encoder
        self.option_model = option_model
        self._output_spec = output_spec
        self.action_spec = action_spec
        self.projection_nets = projection_nets

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
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(
            batch_squash.flatten, observations)

        state, network_state = self.encoder(
            observations,
            step_type=step_type,
            network_state=network_state
        )
        state = self.option_model(state)

        state = batch_squash.unflatten(state)

        def call_projection_net(proj_net):
            distribution, _ = proj_net(
                state, outer_rank, training=training, mask=mask)
            return distribution

        output_actions = tf.nest.map_structure(
            call_projection_net, self.projection_nets)

        return output_actions, network_state
