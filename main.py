import tensorflow as tf
import gym
import os

from layers.gumbel_softmax import GumbelSoftmaxLayer


dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = dir_path+"/checkpoints/"
tf_logdir = dir_path+"/tmp/tf_logdir"
save_checkpoints = False
n_epochs = 10
steps_per_epoch = 10000
batch_size = 64


tf.debugging.experimental.enable_dump_debug_info(
    dump_root=tf_logdir,
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)


def get_states(environment, number_of_episodes):
    env = gym.make(environment)
    n = 0
    while(n < number_of_episodes):
        observation = env.reset()
        done = False
        while(not done):
            yield [observation]
            observation, reward, done, info = env.step(
                env.action_space.sample())
        n += 1


def normalize_data(x):
    xmin = 0
    xmax = 500
    x = tf.math.divide(
        tf.math.subtract(x, xmin),
        (xmax - xmin))
    return ([x], [x])


dataset = tf.data.Dataset.from_generator(
    lambda: get_states('Taxi-v3', 100), output_types=tf.int32, output_shapes=(1,))

dataset = dataset.map(normalize_data)


# for count_batch in dataset.repeat().batch(10).take(10):
#     print(count_batch[0].numpy(), ',', count_batch[1].numpy())

gumbel_layer = GumbelSoftmaxLayer(10, 2, 5.0, 0.7, n_epochs)

encoder = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), batch_size=batch_size),
    tf.keras.layers.Dense(40, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(20),
    gumbel_layer,
    tf.keras.layers.Flatten()
])
# encoder.summary()

decoder = tf.keras.Sequential([
    tf.keras.Input(shape=(20,), batch_size=batch_size),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
# decoder.summary()

state_autoencoder = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), batch_size=batch_size),
    encoder,
    decoder,
])

if (save_checkpoints):
    state_autoencoder.save_weights(checkpoint_path)
    load_status = state_autoencoder.load_weights(checkpoint_path)
    load_status.assert_consumed()  # assert that all model variables have been restored

# for data, label in dataset.take(1):
#     print("Input: ", data)
#     print("Encoding: ", encoder(data).numpy())
#     print("Decoding: ", state_autoencoder(data).numpy())

state_autoencoder.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)

train_dataset = dataset.shuffle(buffer_size=1024).repeat().batch(
    batch_size, drop_remainder=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tf_logdir)

history = state_autoencoder.fit(
    train_dataset, epochs=n_epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=[tensorboard_callback, gumbel_layer.get_update_callback()])
