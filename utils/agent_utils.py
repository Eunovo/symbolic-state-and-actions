import tensorflow as tf

from tf_agents.trajectories.time_step import TimeStep

def get_prepare(spec):
    def fix_tensor(tensor, spec):
        tensor = tf.reshape(tensor, spec.shape)
        return tf.cast(tensor, spec.dtype)

    def prepare(time_step):
        step_type = fix_tensor(
            time_step.step_type, spec.step_type)
        reward = fix_tensor(
            time_step.reward, spec.reward)
        discount = fix_tensor(
            time_step.discount, spec.discount)
        observation = fix_tensor(
            time_step.observation, spec.observation)

        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation
        )

    return prepare


def compute_avg_reward(environment, policy, num_episodes=10, prepare=None):
    def use_prepare_if_set(x): return prepare(x) if (prepare) else x
    total_reward = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(
                use_prepare_if_set(time_step))
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_reward += episode_return

    avg_reward = total_reward / num_episodes
    return avg_reward.numpy()[0]
