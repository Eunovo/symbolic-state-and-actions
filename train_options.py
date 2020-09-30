import tensorflow as tf
import os

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

from models import OptionHierachy
from models import StateAutoEncoder
from environments import StateEncoder
from utils import Normalizer, DataCollector
from utils import compute_avg_reward, get_prepare

env_name = "Taxi-v3"
num_iterations = 20000
initial_collect_steps = 100
collect_steps_per_iteration = 1

batch_size = 64
learning_rate = 1e-3
log_interval = 200

num_eval_episodes = 5
eval_interval = 1000

replay_buffer_max_length = 100000

dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_dir = dir_path+"/checkpoints/option_hierachy/"
policy_save_dir = dir_path+"/saved/option_hierachy/"


def load_env(env_name, encoder_path):
    normalizer = Normalizer(0, 499)
    sae = StateAutoEncoder(
        1, 1,
        12, normalize=True,
        normalizer=normalizer
    )
    sae.use_checkpoints(encoder_path)

    train_py_env = StateEncoder(suite_gym.load(env_name), sae)
    eval_py_env = StateEncoder(suite_gym.load(env_name), sae)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    return (train_env, eval_env)


if __name__ == "__main__":
    train_env, eval_env = load_env(
        env_name, dir_path + '/checkpoints/sae/')

    options_agent = OptionHierachy(
        batch_size,
        train_env.action_spec(),
        train_env.time_step_spec(),
        num_iterations,
        replay_buffer_max_length,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir
    )

    replay_buffer = options_agent.get_replay_buffer()
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2
    )
    iterator = iter(dataset)

    prepare = get_prepare(train_env.time_step_spec())
    data_collector = DataCollector(
        train_env, batch_size, replay_buffer, prepare=prepare)

    # Evaluate the agent's policy once before training.
    avg_reward = compute_avg_reward(
        eval_env, options_agent.policy,
        num_eval_episodes, prepare=prepare
    )
    avg_reward_history = [avg_reward]

    data_collector.collect_data(
        options_agent.collect_policy, initial_collect_steps)

    step = 0
    while (step < num_iterations):
        # Collect a few steps every iteration
        data_collector.collect_data(
            options_agent.collect_policy, collect_steps_per_iteration)

        experience, unused_info = next(iterator)
        train_loss = options_agent.train(experience)

        step = options_agent.get_counter()

        if step % log_interval == 0:
            options_agent.save_checkpoint()
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_reward = compute_avg_reward(
                eval_env, options_agent.policy,
                num_eval_episodes, prepare=prepare
            )
            print('step = {0}: Average Reward = {1}'.format(step, avg_reward))
            avg_reward_history.append(avg_reward)

    options_agent.save_policy(policy_save_dir)
