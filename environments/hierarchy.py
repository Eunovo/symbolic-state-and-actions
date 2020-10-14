import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class LowLevelEnv(py_environment.PyEnvironment):
    def __init__(self, env, low_level_model):
        self.env = env
        self.low_level_model = low_level_model

        self._observation_spec = self.env.time_step_spec().observation
        self._action_spec = self.observation_spec

        num_actions = self.env.action_spec().maximum - \
            self.env.action_spec().minimum + 1
        self.env_actions = range(num_actions)
        self.one_hot = tf.one_hot(
            self.env_actions, num_actions, on_value=1.0, off_value=0.0)

        self.search_depth_limit = 1

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        return self.env.reset()

    def _step(self, goal):
        # action here is a goal state
        current_time_step = self.env.current_time_step()
        # we need to plan here
        plan = self.search(
            current_time_step.observation,
            goal
        )

        if len(plan) == 0:
            # return negative score when goal state is not reachable
            return ts.TimeStep(
                observation=current_time_step.observation,
                step_type=current_time_step.step_type,
                discount=current_time_step.discount,
                reward=tf.cast(-50, tf.float32)
            )

        # accumulate rewards accross several actions
        reward = 0
        time_step = None
        for action in plan:
            time_step = self.env.step(action)
            reward += time_step.reward
            if (time_step.is_last()):
                reward = -50
                break

        return ts.TimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            discount=time_step.discount,
            reward=tf.cast(reward, tf.float32)
        )

    def search(self, current_state, goal, depth=0):
        if (tf.math.reduce_all(tf.equal(goal, current_state))):
            return []

        if (depth > self.search_depth_limit):
            return []

        best_plan = []
        for action in self.env_actions:
            plan = [action]
            one_hot_action = self.one_hot[action]
            action_state = tf.concat([one_hot_action, current_state], 0)
            action_state = tf.reshape(action_state, (1, len(action_state)))
            next_state = self.low_level_model.predict(action_state)[0]
            plan.extend(self.search(next_state, goal, depth + 1))

            if (len(plan) <= len(best_plan)):
                best_plan = plan

        return best_plan


class MasterEnv(py_environment.PyEnvironment):
    def __init__(self, env, action_spec):
        self.env = env
        self.options_policy = None
        self._action_spec = action_spec
        self._observation_spec = self.env.time_step_spec().observation

    def set_options_policy(self, options_policy):
        self.options_policy = options_policy

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        return self.env.reset()

    def _step(self, action):
        current_time_step = self.env.current_time_step()
        options_time_step = ts.TimeStep(
            observation=tf.concat([action, current_time_step.observation], 0),
            step_type=current_time_step.step_type,
            discount=current_time_step.discount,
            reward=current_time_step.reward
        )
        goal = self.options_policy.action(options_time_step)
        return self.env.step(goal.action)


class OptionsEnv(py_environment.PyEnvironment):
    def __init__(self, env, observation_spec):
        self.env = env
        self.master_policy = None
        self._observation_spec = observation_spec
        self._action_spec = self.env.time_step_spec().observation

    def set_master_policy(self, master_policy):
        self.master_policy = master_policy

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        time_step = self.env.reset()
        return self.get_options_time_step(time_step)

    def _step(self, action):
        current_time_step = self.env.step(action)
        return self.get_options_time_step(current_time_step)

    def get_options_time_step(self, current_time_step):
        master_action = self.master_policy.action(current_time_step).action
        return ts.TimeStep(
            observation=tf.concat(
                [master_action, current_time_step.observation], 0),
            step_type=current_time_step.step_type,
            discount=current_time_step.discount,
            reward=current_time_step.reward
        )
