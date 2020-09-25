import tensorflow as tf
import gym
from numpy import array
import numpy as np

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.specs import BoundedTensorSpec, TensorSpec
from tf_agents.utils import common

from models import OptionHierachy
from models import StateAutoEncoder
from environments import StateEncoder

env_name = "Taxi-v3"  # @param {type:"string"}
num_iterations = 20000  # @param {type:"integer"}
initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

replay_buffer_max_length = 100000  # @param {type:"integer"}

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

time_step_spec = TimeStep(
    step_type=TensorSpec(shape=(), dtype=tf.int32),
    reward=TensorSpec(shape=()),
    discount=TensorSpec(shape=()),
    observation=TensorSpec(shape=(1,), dtype=tf.float32)
)


def get_prepare(spec):
    def prepare(time_step):
        step_type = tf.reshape(
            time_step.step_type,
            spec.step_type.shape
        )
        reward = tf.reshape(
            time_step.reward,
            spec.reward.shape
        )
        discount = tf.reshape(
            time_step.discount,
            spec.discount.shape
        )
        observation = tf.reshape(
            observation,
            spec.observation.shape
        )
        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=tf.cast(observation, spec.observation.dtype)
        )

    return prepare


def compute_avg_reward(environment, policy, num_episodes=10, prepare=None):
    use_prepare_if_set = lambda x: prepare(x) if (prepare) else x
    total_reward = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(
                use_prepare_if_set(time_step), collect=True)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_reward += episode_return

    avg_reward = total_reward / num_episodes
    return avg_reward.numpy()[0]


def collect_step(environment, policy, buffer, prepare=None):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step, collect=True)
    next_time_step = environment.step(action_step.action)

    use_prepare_if_set = lambda x: prepare(x) if (prepare) else x

    traj = trajectory.from_transition(
        use_prepare_if_set(time_step),
        action_step,
        use_prepare_if_set(next_time_step)
    )

    # Add trajectory to the replay buffer
    values_batched = tf.nest.map_structure(
        lambda t: tf.stack([t] * batch_size),
        traj
    )
    buffer.add_batch(values_batched)


def collect_data(env, policy, buffer, steps, prepare=None):
    for _ in range(steps):
        collect_step(env, policy, buffer, prepare)


if __name__ == "__main__":
    options_agent = OptionHierachy(
        batch_size,
        train_env.action_spec(),
        time_step_spec,
        num_iterations,
        learning_rate=learning_rate
    )

    replay_buffer = options_agent.get_replay_buffer(
        replay_buffer_max_length
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2
    )
    iterator = iter(dataset)

    prepare = get_prepare(time_step_spec)

    # Evaluate the agent's policy once before training.
    avg_reward = compute_avg_reward(
        eval_env, options_agent,
        num_eval_episodes, prepare=prepare
    )
    avg_reward_history = [avg_reward]

    collect_data(
        train_env, options_agent,
        replay_buffer, initial_collect_steps,
        prepare=prepare
    )

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(
            train_env, options_agent,
            replay_buffer, collect_steps_per_iteration,
            prepare=prepare
        )

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = options_agent.train(experience)

        step = options_agent.get_counter()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_reward = compute_avg_reward(
                eval_env, options_agent,
                num_eval_episodes, prepare=prepare
            )
            print('step = {0}: Average Reward = {1}'.format(step, avg_reward))
            avg_reward_history.append(avg_reward)


# def main(environment, n_episodes, state_ae):
#     # Remember to implement two stage training
#     # and reset option_selector before every training loop

#     env = gym.make(environment)
#     options = [
#         OptionNetwork(name='option_1')
#     ]
#     option_selector = OptionSelector(10)
#     optimizer = tf.keras.optimizers.Adam()

#     for episode in range(n_episodes):
#         total_episode_reward = 0
#         with tf.GradientTape() as tape:
#             observation = env.reset()
#             selected_option = options[
#                 option_selector.select_option(
#                     state_ae.encode_state(observation)
#                 )
#             ]

#             done = False
#             while(not done):
#                 state = state_ae.encode_state(observation)
#                 action_index = selected_option.get_action(state)
#                 should_terminate = selected_option.should_terminate(state)

#                 observation, reward, done, info = env.step(
#                     env.action_space[action_index]
#                 )

#                 total_episode_reward += reward
#                 option_selector.receive_reward(reward, tape, optimizer)
#                 selected_option.receive_reward(reward, tape, optimizer)

#                 if should_terminate:
#                     selected_option = options[
#                         option_selector.select_option(state)
#                     ]

#         if (episode % 200 == 0):
#             print("Episode: {0}, Reward: {1}"
#                   .format(episode + 1, total_episode_reward))


# if __name__ == "__main__":
#     state_ae = None
#     main('Taxi-v3', 100, state_ae)
