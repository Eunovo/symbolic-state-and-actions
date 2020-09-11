import tensorflow as tf
import gym
import os

from layers.gumbel_softmax import GumbelSoftmax


dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = dir_path+"/checkpoints/"
n_epochs = 1000
steps_per_epoch = 10000


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
    lambda: get_states('Taxi-v3', 100), output_types=(tf.int32, tf.int32), output_shapes=((1,), (1,)))

# for count_batch in dataset.repeat().batch(10).take(10):
#     print(count_batch[0].numpy(), ',', count_batch[1].numpy())

encoder = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(2, activation=tf.nn.relu),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    GumbelSoftmax(4, 2, 5.0, 0.7, n_epochs)
])

# encoder.summary()

decoder = tf.keras.Sequential([
    tf.keras.Input(shape=(4)),
    tf.keras.layers.Dense(2, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
])
# decoder.summary()

state_autoencoder = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    encoder,
    decoder,
])

state_autoencoder.save_weights(checkpoint_path)
load_status = state_autoencoder.load_weights(checkpoint_path)
load_status.assert_consumed()  # assert that all model variables have been restored

for data, label in dataset.take(1):
    print("Logits: ", encoder(data).numpy())

# state_autoencoder.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss=tf.keras.losses.MeanSquaredError(),
#     metrics=[tf.keras.metrics.Accuracy()],
# )

# train_dataset = dataset.shuffle(buffer_size=1024).repeat().batch(
#     512, drop_remainder=True)

# history = state_autoencoder.fit(
#     train_dataset, epochs=n_epochs, steps_per_epoch=steps_per_epoch)
