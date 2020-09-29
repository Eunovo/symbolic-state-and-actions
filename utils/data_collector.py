import tensorflow as tf

from tf_agents.trajectories import trajectory

class DataCollector:
    def __init__(self, env, batch_size, buffer, prepare=None):
        self.env = env
        self.batch_size = batch_size
        self.buffer = buffer
        self.prepare = prepare

    def preprocess(self, time_step):
        if (self.prepare):
            return self.prepare(time_step)
        return time_step

    def collect_data(self, policy, n_steps):
        for _ in range(n_steps):
            self.collect_step(policy)

    def collect_step(self, policy):
        batch = []
        while(len(batch) < self.batch_size):
            self.env.reset()

            while True:
                time_step = self.env.current_time_step()
                time_step = self.preprocess(time_step)

                if (time_step.is_last() or (len(batch) >= self.batch_size)):
                    break

                action_step = policy.action(time_step)

                next_time_step = self.env.step(action_step.action)
                next_time_step = self.preprocess(next_time_step)

                traj = trajectory.from_transition(
                    time_step,
                    action_step,
                    next_time_step
                )
                batch.append(tf.nest.flatten(traj))

        values_batched = tf.nest.map_structure(
            lambda i: tf.stack([t[i] for t in batch]),
            tuple(range(len(traj)))
        )
        values_batched = tf.nest.pack_sequence_as(traj, values_batched)
        self.buffer.add_batch(values_batched)
