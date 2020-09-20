import tensorflow as tf

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import value_network
from tf_agents.agents.ppo import ppo_agent

from models.hierachy_actor_network import HierachyActorNetwork

class OptionHierachy():
    def __init__(
        self, batch_size,
        action_spec,
        time_step_spec,
        n_iterations,
        learning_rate=1e-3
    ):
        self.batch_size = batch_size
        observation_spec = time_step_spec.observation

        actor_net = HierachyActorNetwork(
            observation_spec,
            action_spec,
            n_iterations,
            n_options=4
        )
        value_net = value_network.ValueNetwork(
            observation_spec,
            fc_layer_params=(100,)
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.agent = ppo_agent.PPOAgent(
            time_step_spec,
            action_spec,
            actor_net=actor_net,
            value_net=value_net,
            optimizer=optimizer,
            normalize_rewards=True,
            train_step_counter=tf.Variable(0)
        )
        
    def get_counter(self):
        return self.agent.train_step_counter.numpy()

    def get_replay_buffer(self, replay_buffer_max_length):
        return tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.batch_size,
            max_length=replay_buffer_max_length
        )

    def action(self, time_step, collect=False):
        if collect:
            return self.agent.collect_policy.action(time_step)
        return self.agent.policy.action(time_step)

    def train(self, experience):
        return self.agent.train(experience).loss

