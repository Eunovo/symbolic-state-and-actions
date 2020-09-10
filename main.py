import tensorflow as tf
import gym


def get_states(environment, number_of_episodes):
    env = gym.make(environment)
    n = 0
    while(n < number_of_episodes):
        observation = env.reset()
        done = False
        while(not done):
            yield ([observation], [observation])
            observation, reward, done, info = env.step(
                env.action_space.sample())
        n += 1


dataset = tf.data.Dataset.from_generator(
    lambda: get_states('Taxi-v3', 1), output_types=(tf.int32, tf.int32), output_shapes=((1,), (1,)) )

# for count_batch in dataset.repeat().batch(10).take(10):
#     print(count_batch[0].numpy(), ',', count_batch[1].numpy())

encoder = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Dense(4)
])
# encoder.summary()

decoder = tf.keras.Sequential([
    tf.keras.Input(shape=(4)),
    tf.keras.layers.Dense(1, activation='relu')
])
# decoder.summary()

state_autoencoder = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    encoder,
    decoder,
])

# for data, label in dataset.take(1):
#     print("Logits: ", state_autoencoder(data).numpy())

state_autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.Accuracy()],
)

history = state_autoencoder.fit(dataset, epochs=100)
