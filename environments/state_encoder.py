import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class StateEncoder(py_environment.PyEnvironment):
    def __init__(self, env, state_encoder):
        self.env = env
        self.state_encoder = state_encoder

        self._action_spec = self.env.action_spec()
        n_bits = self.state_encoder.n_encode_bits
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(n_bits, ), dtype=np.float32,
            minimum=0, maximum=1, name='observation'
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def transform(self, time_step):
        spec = self.time_step_spec()

        state = tf.reshape(time_step.observation, (1,))
        observation = self.state_encoder.encode(state)

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

        return ts.TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation
        )

    def _reset(self):
        time_step = self.env.reset()
        return self.transform(time_step)

    def _step(self, action):
        time_step = self.env.step(action)
        return self.transform(time_step)
