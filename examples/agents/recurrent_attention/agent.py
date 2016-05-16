import sys
import os
import gym
import time
import skimage.io # for some reason this needs to be loaded before tf
import tensorflow as tf
import numpy as np
import random

import resnet as resnet

from tensorflow.models.rnn.rnn_cell import GRUCell

num_categories = 1000 # using the first 10 categories of imagenet for now.
num_directional_actions = 4 # up, right, down, left
num_zoom_actions = 2 # zoom in, zoom out
num_actions = num_categories + num_directional_actions + num_zoom_actions

DQN_GAMMA = 0.99
MOVING_AVERAGE_DECAY = 0.9
REPLAY_MEMEORY_SIZE = 1000000

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('continue', False, 'resume from latest saved state')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('num_episodes', 100000, 'number of epsidoes to run')
tf.app.flags.DEFINE_integer('glimpse_size', 64, '64 or 96 or 224')
tf.app.flags.DEFINE_integer('hidden_size', 1024, '')
tf.app.flags.DEFINE_integer('max_episode_steps', 10, '')
tf.app.flags.DEFINE_string('train_dir', '/tmp/agent_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('restore_resnet', '', 'path to resnet ckpt to restore from')


class Episode:
    obvs = None # Set in done()
    actions = []
    rewards = []
    step = 0
    _done = False

    @property
    def num_frames(self):
        assert self._done
        return self.obvs.shape[0]

    @property
    def num_actions(self):
        assert self._done
        return self.step

    def done(self, obvs):
        self._done = True
        self.obvs = obvs[0:self.step+1,:]

    def store(self, action, reward):
        self.actions.append(action)
        self.rewards.append(reward)
        self.step += 1

class Agent(object):
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    dqn_epsilon = 0.1
    replay_memory = [] # Filled with Episode instances

    def __init__(self, sess):
        self.sess = sess
        self.global_step = tf.get_variable('global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        self.observations = None

        self._build()
        self._setup_train()


    def _setup_train(self):
        batchnorm_updates = tf.get_collection(resnet.UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)

        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        apply_gradient_op = optimizer.apply_gradients(grads, global_step=self.global_step)

        self.train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)

        self.saver = tf.train.Saver(tf.all_variables())

        if FLAGS.__getattr__('continue'):
            latest = tf.train.latest_checkpoint(FLAGS.train_dir)
            if not latest:
                print "No checkpoint to continue from in", FLAGS.train_dir
                sys.exit(1)
            print "continue", latest
            self.saver.restore(self.sess, latest)

        if len(FLAGS.restore_resnet) > 0:
            print "restoring resnet..."
            resnet_variables_to_restore = tf.get_collection(resnet.RESNET_VARIABLES)
            saver = tf.train.Saver(resnet_variables_to_restore)
            # '/Users/ryan/src/tensorflow-resnet/ResNet-L50.ckpt'
            saver.restore(self.sess, FLAGS.restore_resnet)
            print "done"

    def _build(self):
        self.is_training = tf.placeholder('bool', [], name='is_training')
        self.is_terminal = tf.placeholder('bool', [], name='is_terminal')
        self.inputs = tf.placeholder('float', [None, FLAGS.glimpse_size, FLAGS.glimpse_size, 3], name='inputs')
        self.last_reward = tf.placeholder('float', [], name='last_reward')
        self.last_action = tf.placeholder('int32', [], name='last_action')

        # CNN
        # first axis of inputs is time. conflate with batch in cnn.
        # batch size is always 1 with this agent.
        x = self.inputs
        x = resnet.inference_small(x, is_training=self.is_training,
                                   num_blocks=3)
        # add batch dimension. output: batch=1, time, height, width, depth=3
        x = tf.expand_dims(x, 0)
        # END CNN

        self.cell = GRUCell(FLAGS.hidden_size)
        x, states = tf.nn.dynamic_rnn(self.cell, x, dtype='float')

        assert states.get_shape().as_list() == [None, self.cell.state_size]
        assert x.get_shape().as_list() == [1,  None, self.cell.output_size]
        x = tf.squeeze(x, squeeze_dims=[0]) # remove first axis

        action_values = tf.contrib.layers.fully_connected(x, num_actions,
            weight_regularizer=tf.contrib.layers.l2_regularizer(0.00004),
            name='action_values')

        print "action_values", action_values.get_shape()

        def action_value_slice(index):
          y = tf.expand_dims(index, 0)
          z = tf.zeros([1], dtype='int32')
          begin = tf.concat(0, [y, z]) # WAY TOO HARD. FILE A BUG
          #print 'begin shape', begin.get_shape()
          s = tf.slice(action_values, begin, [1, num_actions])
          return tf.squeeze(s)

        input_size = tf.shape(self.inputs)[0]
        last_values = action_value_slice(input_size - 1)

        #last_action_value = action_values[last_index] # PREFERED WAY TO DO IT.

        print "last_value.get_shape()", last_values.get_shape().as_list()
        assert last_values.get_shape().as_list() == [num_actions]

        last_max_value = tf.reduce_max(last_values)
        self.last_maximizing_action = tf.argmax(last_values, 0)

        second_last_values = action_value_slice(input_size - 2)
        inferred_future_reward = tf.squeeze(
            tf.slice(second_last_values,
                     tf.expand_dims(self.last_action, 0),
                     [1]))

        def terminal():
            return self.last_reward

        def not_teriminal():
            return self.last_reward + DQN_GAMMA * last_max_value

        expected_future_reward = tf.cond(self.is_terminal, terminal, not_teriminal)

        q_loss = tf.square(expected_future_reward - inferred_future_reward, name='q_loss')

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([q_loss] + regularization_losses, name='loss')
        tf.scalar_summary('loss', self.loss)

        # loss_avg
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
        tf.add_to_collection(resnet.UPDATE_OPS_COLLECTION, ema.apply([self.loss]))
        loss_avg = ema.average(self.loss)
        tf.scalar_summary('loss_avg', loss_avg)


    def _build_action(self, x, name, num_possible_actions, labels):
        return prob, loss

    @property
    def current_ep(self):
        return self.replay_memory[len(self.replay_memory)-1]

    def act(self, observation):
        #print "observation mean", np.mean(observation)
        #print "observation shape", observation.shape
        assert observation.shape == (FLAGS.glimpse_size, FLAGS.glimpse_size, 3)

        # With probability dqn_epsilon select a random action.
        if random.random() < self.dqn_epsilon:
            action = np.random.randint(0, num_actions)
            return action

        # Otherwise select an action that maximizes Q
        i = self.current_ep.step
        assert i < FLAGS.max_episode_steps
        self.observations[i, :] = observation
        obvs = self.observations[:i+1,:]
        action = self.sess.run(self.last_maximizing_action, {
           self.is_training: False,
           self.inputs: obvs,
        })

        return action

    def reset(self, observation):
        self.observations = np.zeros((FLAGS.max_episode_steps, FLAGS.glimpse_size, FLAGS.glimpse_size, 3))
        self.observations[0,:] = observation

        self.replay_memory.append(Episode())

        # Memory limit on replay_memory
        if len(self.replay_memory) > REPLAY_MEMEORY_SIZE:
            self.replay_memory.pop(0)

    def store(self, observation, action, reward, done, correct_answer):
        ep = self.current_ep
        ep.store(action, reward)
        self.observations[:ep.step, :] = observation

        if done or ep.step >= FLAGS.max_episode_steps:
            ep.done(self.observations)
            self.observations = None
            print "episode done. num_actions %d num_frames %d" % (ep.num_actions, ep.num_frames)


    def train(self):
        step = self.sess.run(self.global_step)
        write_summary = (step % 10 == 0 and step > 1)
        # Sample random minibatch of transititons
        if len(self.replay_memory) < 10: return # skip train

        random_index = np.random.randint(0, len(self.replay_memory) - 1)
        random_episode = self.replay_memory[random_index]

        assert random_episode._done

        random_frame = np.random.randint(1, random_episode.num_frames)

        is_terminal = (random_frame == random_episode.num_frames - 1)
        last_reward = random_episode.rewards[random_frame]
        last_action = random_episode.actions[random_frame]

        obvs = random_episode.obvs[0:random_frame+1]

        i = [self.train_op, self.loss]

        if write_summary:
            i.append(self.summary_op)

        o = self.sess.run(i, {
            self.is_training: True,
            self.is_terminal: is_terminal,
            self.inputs: obvs, 
            self.last_reward: last_reward,
            self.last_action: last_action,
        })

        loss_value = o[1]

        if write_summary:
            summary_str = o[2]
            self.summary_writer.add_summary(summary_str, step)

        print "step %d: %f loss" % (step, loss_value)

        if step % 1000 == 0 and step > 0:
            print 'save checkpoint'
            self.saver.save(self.sess, self.checkpoint_path, global_step=self.global_step)

def main(_):
    env = gym.make('Attention%d-v0' % FLAGS.glimpse_size)

    #env.monitor.start('/tmp/attention', force=True)

    sess = tf.Session(config=tf.ConfigProto(
                      allow_soft_placement=True,
                      log_device_placement=False))

    agent = Agent(sess)

    sess.run(tf.initialize_all_variables())

    for i_episode in xrange(1, FLAGS.num_episodes):
        observation = env.reset()

        agent.reset(observation)

        for t in xrange(FLAGS.max_episode_steps):
            env.render()
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            agent.store(observation, action, reward, done, info)
            agent.train()
            if done: break

    #env.monitor.close()

if __name__ == '__main__':
    tf.app.run()
