import tensorflow as tf
import gym

from models import OptionNetwork, OptionSelector


def main(environment, n_episodes, state_ae):
    # Remember to implement two stage training
    # and reset option_selector before every training loop

    env = gym.make(environment)
    options = [
        OptionNetwork(name='option_1')
    ]
    option_selector = OptionSelector(10)
    optimizer = tf.keras.optimizers.Adam()

    for episode in range(n_episodes):
        total_episode_reward = 0
        with tf.GradientTape() as tape:
            observation = env.reset()
            selected_option = options[
                option_selector.select_option(
                    state_ae.encode_state(observation)
                )
            ]

            done = False
            while(not done):
                state = state_ae.encode_state(observation)
                action_index = selected_option.get_action(state)
                should_terminate = selected_option.should_terminate(state)

                observation, reward, done, info = env.step(
                    env.action_space[action_index]
                )

                total_episode_reward += reward
                option_selector.receive_reward(reward, tape, optimizer)
                selected_option.receive_reward(reward, tape, optimizer)

                if should_terminate:
                    selected_option = options[
                        option_selector.select_option(state)
                    ]

        if (episode % 200 == 0):
            print("Episode: {0}, Reward: {1}"
                  .format(episode + 1, total_episode_reward))


if __name__ == "__main__":
    state_ae = None
    main('Taxi-v3', 100, state_ae)
