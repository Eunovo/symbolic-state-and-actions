import tensorflow as tf

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.policies import policy_saver
from tf_agents.policies import greedy_policy, actor_policy
from tf_agents.utils import common

from models.hierachy_actor_network import HierachyActorNetwork


class OptionHierachy():
    def __init__(
        self, batch_size,
        action_spec,
        time_step_spec,
        n_iterations,
        replay_buffer_max_length,
        learning_rate=1e-3,
        checkpoint_dir=None
    ):
        self.batch_size = batch_size
        self.time_step_spec = time_step_spec
        self.action_spec = action_spec
        observation_spec = self.time_step_spec.observation

        self.actor_net = HierachyActorNetwork(
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
        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        self.agent = ppo_agent.PPOAgent(
            time_step_spec,
            self.action_spec,
            actor_net=self.actor_net,
            value_net=value_net,
            optimizer=optimizer,
            normalize_rewards=True,
            normalize_observations=False,
            train_step_counter=self.global_step
        )
        self.agent.initialize()
        self.agent.train = common.function(self.agent.train)

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.batch_size,
            max_length=replay_buffer_max_length
        )

        self.train_checkpointer = None
        if (checkpoint_dir):
            self.train_checkpointer = common.Checkpointer(
                ckpt_dir=checkpoint_dir,
                max_to_keep=1,
                agent=self.agent,
                policy=self.agent.policy,
                replay_buffer=self.replay_buffer,
                global_step=self.global_step
            )
            self.train_checkpointer.initialize_or_restore()

        self.policy_saver = policy_saver.PolicySaver(self.agent.policy)

    def get_counter(self):
        return self.agent.train_step_counter.numpy()

    def get_replay_buffer(self):
        return self.replay_buffer

    def get_option_policies(self):
        return [
            greedy_policy.GreedyPolicy(
                actor_policy.ActorPolicy(
                    time_step_spec=self.time_step_spec,
                    action_spec=self.action_spec,
                    actor_network=option_net
                )
            )
            for option_net in self.actor_net.get_options()
        ]

    def action(self, time_step, collect=False):
        if collect:
            return self.agent.collect_policy.action(time_step)
        return self.agent.policy.action(time_step)

    def train(self, experience):
        return self.agent.train(experience).loss

    def save_checkpoint(self):
        if (self.train_checkpointer == None):
            raise ValueError(
                'Cannot use checkpoint if checkpoint dir is not defined')

        self.train_checkpointer.save(self.global_step)

    def save_policy(self, save_dir):
        self.policy_saver.save(save_dir)
