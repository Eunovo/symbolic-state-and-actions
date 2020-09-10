import tensorflow as tf
import gym


def get_states(environment, number_of_episodes):
    print(environment, number_of_episodes)
    env = gym.make(environment)
    n = 0
    while(n < number_of_episodes):
        observation = env.reset()
        done = False
        while(not done):
            yield observation
            observation, reward, done, info = env.step(
                env.action_space.sample())
        n += 1


dataset = tf.data.Dataset.from_generator(
    lambda: get_states('Taxi-v3', 1), output_types=tf.int32, output_shapes=(), )

# for count_batch in dataset.repeat().batch(10).take(10):
#     print(count_batch.numpy())
