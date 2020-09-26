import tensorflow as tf
import gym
import os

from models import StateAutoEncoder
from utils import Normalizer

dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = dir_path+"/checkpoints/sae/"
tf_logdir = dir_path+"/tmp/tf_logdir/"
model_save_dir = dir_path+"/saved/sae/"
save_checkpoints = True
n_epochs = 10
steps_per_epoch = 10000
batch_size = 128


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


def generate_dataset(environment, number_of_episodes, normalizer):
    dataset = tf.data.Dataset.from_generator(
        lambda: get_states('Taxi-v3', number_of_episodes),
        output_types=tf.int32, output_shapes=(1,)
    )

    def normalize(x):
        x = normalizer.normalize(x)
        return ([x], [x])

    return dataset.map(normalize)


def parse_args(save_checkpoints, checkpoint_path, model_save_dir):
    import argparse

    parser = argparse.ArgumentParser(
        description='Train the sate auto encoder.')
    parser.add_argument(
        '-save-checkpoints',
        help='Save model training checkpoints',
        default=save_checkpoints
    )
    parser.add_argument(
        '-board',
        help='Use tensorboard',
        default=False
    )
    parser.add_argument(
        '--savedir',
        help='Model save path',
        default=model_save_dir
    )
    parser.add_argument(
        '--checkpointdir',
        help='Save dir for model checkpoints',
        default=checkpoint_path
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(save_checkpoints, checkpoint_path, model_save_dir)

    callbacks = []
    if (args.board):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tf_logdir)
        callbacks.append(tensorboard_callback)

    normalizer = Normalizer(0, 499)

    dataset = generate_dataset('Taxi-v3', 100, normalizer)

    state_autoencoder = StateAutoEncoder(
        n_epochs, steps_per_epoch,
        12, normalize=True,
        normalizer=normalizer
    )

    if (args.save_checkpoints):
        state_autoencoder.use_checkpoints(args.checkpointdir)

    state_autoencoder.compile()

    train_dataset = dataset.shuffle(buffer_size=1024).repeat().batch(
        batch_size, drop_remainder=True)
    history = state_autoencoder.fit(train_dataset, callbacks)
    state_autoencoder.save(args.savedir)
    
