import tensorflow as tf
import numpy as np
import os

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks import value_network, actor_distribution_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.policies import policy_saver
from tf_agents.utils import common
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from environments import StateEncoder
from environments import LowLevelEnv, MasterEnv, OptionsEnv
from models import StateAutoEncoder
from models.hierarchy import OptionsNetwork
from train_actions import setup_model
from utils import Normalizer


num_state_bits = 12
learning_rate = 0.01
num_options = 4
env_name = 'Taxi-v3'

batch_size = 64
replay_buffer_max_length = 100000
warmup_period = 1000
joint_update_period = 1000
num_iterations = 20000
log_interval = 1000
eval_interval = 1000
checkpoint_interval = 1000

dir_path = os.path.dirname(os.path.realpath(__file__))
save_dir = dir_path+'/saved/hierarchy-v2/'
checkpoint_dir = dir_path+'/checkpoints/hierarchy-v2/'
encoder_path = dir_path+'/checkpoints/sae/'
low_level_model_path = dir_path+'/checkpoints/low_level_actions/'

master_collect_episodes = 1
options_collect_episodes = 1


def load_env(env_name, sae):
    train_py_env = StateEncoder(suite_gym.load(env_name), sae)
    eval_py_env = StateEncoder(suite_gym.load(env_name), sae)

    train_env = train_py_env
    eval_env = eval_py_env

    return (train_env, eval_env)


def create_replay_buffer(agent, batch_size, max_length):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=batch_size,
        max_length=max_length
    )


def create_train_checkpointer(
    checkpoint_dir,
    agent,
    replay_buffer,
    step
):
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=step
    )
    train_checkpointer.initialize_or_restore()
    return train_checkpointer


def populate_buffer(env, replay_buffer, policy, num_episodes, batch_size):
    def add_to_replay_buffer(traj):
        traj_batched = tf.nest.map_structure(
            lambda t: tf.stack([t] * batch_size), traj)
        replay_buffer.add_batch(traj_batched)

    observers = [add_to_replay_buffer]
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env, policy, observers, num_episodes=num_episodes)

    # Initial driver.run will reset the environment and initialize the policy.
    driver.run()


normalizer = Normalizer(0, 499)
sae = StateAutoEncoder(
    1, 1,
    num_state_bits, normalize=True,
    normalizer=normalizer
)
sae.use_checkpoints(encoder_path)

train_env, _ = load_env(env_name, sae)

master_action_spec = array_spec.BoundedArraySpec(
    shape=((num_options,)), dtype=np.float32,
    minimum=0, maximum=1, name='master_action'
)

options_observation_spec = array_spec.BoundedArraySpec(
    shape=((num_options + num_state_bits), ), dtype=np.float32,
    minimum=0, maximum=1, name='option_observation'
)
options_time_step_spec = ts.TimeStep(
    step_type=train_env.time_step_spec().step_type,
    reward=train_env.time_step_spec().reward,
    discount=train_env.time_step_spec().discount,
    observation=options_observation_spec
)

num_actions = train_env.action_spec().maximum - train_env.action_spec().minimum + 1
low_level_model, callbacks = setup_model(
    num_actions, num_state_bits, sae, low_level_model_path)

low_level_env = LowLevelEnv(train_env, low_level_model)

options_env = OptionsEnv(
    low_level_env, options_observation_spec)
option_train_env = tf_py_environment.TFPyEnvironment(options_env)

master_env = MasterEnv(
    low_level_env, master_action_spec)
master_train_env = tf_py_environment.TFPyEnvironment(master_env)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
global_step = tf.compat.v1.train.get_or_create_global_step()

master_value_network = value_network.ValueNetwork(
    master_train_env.time_step_spec().observation,
    fc_layer_params=(100,)
)

master_actor_network = actor_distribution_network.ActorDistributionNetwork(
    master_train_env.time_step_spec().observation,
    master_train_env.action_spec(),
    fc_layer_params=(100,)
)

master_agent = ppo_agent.PPOAgent(
    master_train_env.time_step_spec(),
    master_train_env.action_spec(),
    optimizer=optimizer,
    actor_net=master_actor_network,
    value_net=master_value_network,
    train_step_counter=tf.Variable(0)
)
master_agent.initialize()
master_agent.train = common.function(master_agent.train)
options_env.set_master_policy(master_agent.policy)

options_actor_network = OptionsNetwork(
    option_train_env.time_step_spec().observation,
    option_train_env.action_spec(),
    num_options
)

options_value_network = value_network.ValueNetwork(
    option_train_env.time_step_spec().observation,
    fc_layer_params=(100,)
)

options_agent = ppo_agent.PPOAgent(
    option_train_env.time_step_spec(),
    option_train_env.action_spec(),
    optimizer=optimizer,
    actor_net=options_actor_network,
    value_net=options_value_network,
    train_step_counter=tf.Variable(0)
)
options_agent.initialize()
options_agent.train = common.function(options_agent.train)
master_env.set_options_policy(options_agent.policy)


master_rb = create_replay_buffer(
    master_agent, batch_size, replay_buffer_max_length)
options_rb = create_replay_buffer(
    options_agent, batch_size, replay_buffer_max_length)

master_ds = master_rb.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2
)
master_iter = iter(master_ds)
options_ds = options_rb.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2
)
options_iter = iter(options_ds)

master_checkpointer = create_train_checkpointer(
    checkpoint_dir + "master/", master_agent, master_rb, global_step)
options_checkpointer = create_train_checkpointer(
    checkpoint_dir + "options/", options_agent, options_rb, global_step)

master_saver = policy_saver.PolicySaver(master_agent.policy)
options_saver = policy_saver.PolicySaver(options_agent.policy)


def check_interval(interval):
    return global_step % interval == 0


while (global_step < num_iterations):
    for _ in range(warmup_period):
        populate_buffer(
            master_train_env, master_rb,
            master_agent.collect_policy,
            master_collect_episodes,
            batch_size
        )
        experience, unused_info = next(master_iter)
        master_loss = master_agent.train(experience)

    for _ in range(joint_update_period):
        populate_buffer(
            master_train_env, master_rb,
            master_agent.collect_policy,
            master_collect_episodes,
            batch_size
        )
        populate_buffer(
            option_train_env, options_rb,
            options_agent.collect_policy,
            options_collect_episodes,
            batch_size
        )
        option_exp, unused_info = next(options_iter)
        options_loss = options_agent.train(option_exp)
        master_exp, unused_info = next(master_iter)
        master_loss = master_agent.train(master_exp)

    if check_interval(log_interval):
        print(
            'step = {0}: master loss = {1}, options loss = {2}'.format(
                global_step, master_loss, options_loss)
        )

    if check_interval(checkpoint_interval):
        master_checkpointer.save(global_step)
        options_checkpointer.save(global_step)
        print('Checkpoint saved!')

    # Reset master here

master_saver.save(save_dir + "master/")
options_saver.save(save_dir + "options/")
print("Policies Saved!")
