import tensorflow as tf
import numpy as np
import gym
import os

from models import StateAutoEncoder
from models.hierarchy import ActionNetwork
from utils import Normalizer

dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_dir = dir_path+"/checkpoints/low_level_actions/"
model_save_dir = dir_path+"/saved/low_level_actions/"
sae_path = dir_path+"/checkpoints/sae/"

n_epochs = 1000
steps_per_epoch = 100
num_state_bits = 12
env_name = "Taxi-v3"
batch_size = 64
num_collect_episodes = 100


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def create_data_generator(env, num_collect_episodes, encoder, get_action_code):
    n = 0
    while(n < num_collect_episodes):
        observation = env.reset()
        done = False
        while(not done):
            prev_observation = encoder.encode([observation])[0]
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            action_code = get_action_code(action)
            yield (
                tf.concat([action_code, prev_observation], 0),
                encoder.encode([observation])[0]
            )
        n += 1


def create_dataset_from_generator(generator):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32), output_shapes=((18,), (12,))
    )
    return dataset


def setup_env(env_name):
    env = gym.make(env_name)
    num_actions = env.action_space.n
    actions = np.array(range(num_actions))
    one_hot = get_one_hot(actions, num_actions)
    dataset = create_dataset_from_generator(
        lambda: create_data_generator(
            env, num_collect_episodes, sae, lambda x: one_hot[x])
    )

    train_ds = dataset.shuffle(buffer_size=1024).repeat().batch(
        batch_size, drop_remainder=True)

    return num_actions, num_state_bits, train_ds


def setup_model(num_actions, num_state_bits, sae, checkpoint_dir):
    low_level_action_model = ActionNetwork(
        num_actions, num_state_bits, n_epochs)
    low_level_action_model.compile(
        optimizer='adam',
        loss='MSE',
        metrics=['accuracy'],
    )

    checkpoint_file = checkpoint_dir + "ckpt"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_weights_only=True
    )

    if (len(os.listdir(checkpoint_dir)) > 0):
        print("Restoring from", checkpoint_file)
        low_level_action_model.load_weights(checkpoint_file)

    return low_level_action_model, [
        checkpoint_callback,
        low_level_action_model.gumbel_callback
    ]


if __name__ == "__main__":
    normalizer = Normalizer(0, 499)
    sae = StateAutoEncoder(
        1, 1,
        num_state_bits, normalize=True,
        normalizer=normalizer
    )
    sae.use_checkpoints(sae_path)

    num_actions, num_state_bits, train_ds = setup_env(env_name)

    low_level_action_model, callbacks = setup_model(
        num_actions,
        num_state_bits,
        sae,
        checkpoint_dir
    )

    low_level_action_model.fit(
        train_ds,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )

    low_level_action_model.save(model_save_dir)
