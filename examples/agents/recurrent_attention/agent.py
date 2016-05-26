from skimage.io import imread # for some reason this needs to be loaded before tf

import sys
import os
import errno
import gym
import time
import skimage.io
import tensorflow as tf
import numpy as np
import random

import resnet as resnet
from replay_memory import ReplayMemory

from tensorflow.models.rnn.rnn_cell import GRUCell
from gym.envs.attention.common import *

DQN_GAMMA = 0.99
MOVING_AVERAGE_DECAY = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('resume', False, 'resume from latest saved state')
tf.app.flags.DEFINE_boolean('use_rnn', False, 'use a rnn to train the network')
tf.app.flags.DEFINE_boolean('show_train_window', False, 'show the training window')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('num_episodes', 100000, 'number of epsidoes to run')
tf.app.flags.DEFINE_integer('glimpse_size', 32, '32 or 64')
tf.app.flags.DEFINE_integer('hidden_size', 1024, '')
tf.app.flags.DEFINE_integer('max_episode_steps', 10, '')
tf.app.flags.DEFINE_string('train_dir', '/tmp/agent_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('restore_resnet', '', 'path to resnet ckpt to restore from')


DEBUG = False

def log(msg):
    if DEBUG: print msg

class Episode:
    def __init__(self, first_observation_params):
        self.actions = []
        self.rewards = []
        self.step = 0
        self._done = False
        self._obvs_set = False
        self.img_fn = first_observation_params[0]
        self.y_params = [ first_observation_params[1] ]
        self.x_params = [ first_observation_params[2] ]
        self.zoom_params = [ first_observation_params[3] ]

    @property
    def obvs(self):
        assert self._done
        if self._obvs_set:
            return self._obvs

        # TODO need data_dir
        data_dir = "/Users/ryan/data/imagenet-small/imagenet-small-train"

        full_img_fn = os.path.join(data_dir, self.img_fn)
        img = imread(full_img_fn)
        if len(img.shape) == 2:
            img = np.dstack([img, img, img])

        self._obvs = np.zeros((self.num_frames, FLAGS.glimpse_size, FLAGS.glimpse_size, 3))
        for i in range(self.num_frames):
            frame = make_observation(img=img,
                glimpse_size=FLAGS.glimpse_size,
                y=self.y_params[i],
                x=self.x_params[i],
                zoom=self.zoom_params[i])
            self._obvs[i,:] = frame

        self._obvs_set = True
        return self._obvs


    @property
    def num_frames(self):
        assert self._done
        return len(self.y_params)

    @property
    def num_actions(self):
        assert self._done
        return self.step

    def done(self):
        self._done = True

    def store(self, observation_params, action, reward):
        assert not self._done
        assert observation_params[0] == self.img_fn
        self.actions.append(action)
        self.rewards.append(reward)
        self.y_params.append(observation_params[1])
        self.x_params.append(observation_params[2])
        self.zoom_params.append(observation_params[3])
        self.step += 1

class Agent(object):
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    replay_memory_path = os.path.join(FLAGS.train_dir, 'replay_memory')

    def __init__(self, sess):
        self._maybe_delete()

        self.replay_memory = ReplayMemory(self.replay_memory_path)
        self.current_ep = None # set in reset()

        self.sess = sess

        self.global_step = tf.get_variable('global_step', [], dtype='int32',
            initializer=tf.constant_initializer(0), trainable=False)
        self.val_step = tf.get_variable('val_step', [], dtype='int32',
            initializer=tf.constant_initializer(0), trainable=False)

        self._build()
        self._setup_train()

    def _maybe_delete(self):
        # If we aren't resuming and the train dir exists prompt to delete
        if not FLAGS.resume and os.path.isdir(FLAGS.train_dir):
            print "Starting a new training session but %s exists" % FLAGS.train_dir
            sys.stdout.write("Do you want to delete all the files and recreate it [y] ")
            response = raw_input().lower()
            if response == "" or response == "y" or response == "yes":
                import shutil
                print "rm -rf %s" % FLAGS.train_dir
                shutil.rmtree(FLAGS.train_dir)

    def _setup_train(self):
        batchnorm_updates = tf.get_collection(resnet.UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)

        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        for grad, var in grads:
            tf.histogram_summary(var.op.name, var)
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        apply_gradient_op = optimizer.apply_gradients(grads, global_step=self.global_step)

        self.train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)

        self.saver = tf.train.Saver(tf.all_variables())

        self.sess.run(tf.initialize_all_variables())

        if FLAGS.resume:
            latest = tf.train.latest_checkpoint(FLAGS.train_dir)
            if not latest:
                print "No checkpoint to resume from in", FLAGS.train_dir
                sys.exit(1)
            print "resume", latest
            print "replay memory size:", self.replay_memory.count()
            self.saver.restore(self.sess, latest)

        if len(FLAGS.restore_resnet) > 0:
            print "restoring resnet..."
            resnet_variables_to_restore = tf.get_collection(resnet.RESNET_VARIABLES)
            saver = tf.train.Saver(resnet_variables_to_restore)
            # /Users/ryan/src/tensorflow-resnet/tensorflow-resnet-pretrained-20160509/ResNet-L50.ckpt
            saver.restore(self.sess, FLAGS.restore_resnet)
            print "done"

    def _build(self):
        self.is_training = tf.placeholder('bool', [], name='is_training')
        self.is_terminal = tf.placeholder('bool', [], name='is_terminal')
        self.inputs = tf.placeholder('float', [None, FLAGS.glimpse_size, FLAGS.glimpse_size, 3], name='inputs')
        self.reward = tf.placeholder('float', [], name='reward')
        self.action = tf.placeholder('int32', [], name='action')

        # CNN
        # first axis of inputs is time. conflate with batch in cnn.
        # batch size is always 1 with this agent.
        x = self.inputs
        x = resnet.inference_small(x,
                                   is_training=self.is_training,
                                   num_classes=None,
                                   use_bias=True,
                                   num_blocks=1)

        if FLAGS.use_rnn:
            # add batch dimension. output: batch=1, time, height, width, depth=3
            x = tf.expand_dims(x, 0)
            self.cell = GRUCell(FLAGS.hidden_size)
            x, states = tf.nn.dynamic_rnn(self.cell, x, dtype='float')

            assert states.get_shape().as_list() == [None, self.cell.state_size]
            assert x.get_shape().as_list() == [1,  None, self.cell.output_size]
            x = tf.squeeze(x, squeeze_dims=[0]) # remove first axis

        q_values = tf.contrib.layers.fully_connected(x, num_actions,
            weight_regularizer=tf.contrib.layers.l2_regularizer(0.00004),
            name='q_values')

        #print "q_values", q_values.get_shape()

        def action_value_slice(index):
          # First need to build the begin argument to tf.slice.
          # It should be begin=[index, 0] but TF makes this
          # WAY TOO HARD. TODO FILE A BUG
          y = tf.expand_dims(index, 0)
          z = tf.zeros([1], dtype='int32')
          begin = tf.concat(0, [y, z])

          s = tf.slice(q_values, begin, [1, num_actions])
          return tf.squeeze(s)

        input_size = tf.shape(self.inputs)[0]
        last_values = action_value_slice(input_size - 1)

        # PREFERED WAY TO DO IT.
        #last_index = input_size - 1
        #last_action_value = q_values[last_index]

        #print "last_value.get_shape()", last_values.get_shape().as_list()
        assert last_values.get_shape().as_list() == [num_actions]

        last_max_value = tf.reduce_max(last_values)
        self.last_maximizing_action = tf.argmax(last_values, 0)

        second_last_values = action_value_slice(input_size - 2)
        inferred_future_reward = tf.squeeze(
            tf.slice(second_last_values,
                     tf.expand_dims(self.action, 0),
                     [1]))

        def terminal():
            return self.reward

        def not_teriminal():
            return self.reward + DQN_GAMMA * last_max_value

        expected_future_reward = tf.cond(self.is_terminal, terminal, not_teriminal)

        q_loss = tf.square(expected_future_reward - inferred_future_reward, name='q_loss')

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([q_loss] + regularization_losses, name='loss')
        tf.scalar_summary('loss', self.loss)

        # loss_avg
        ema = tf.train.ExponentialMovingAverage(0.999, self.global_step)
        tf.add_to_collection(resnet.UPDATE_OPS_COLLECTION, ema.apply([self.loss]))
        loss_avg = ema.average(self.loss)
        tf.scalar_summary('loss_avg', loss_avg)

        # validation error avg
        ema = tf.train.ExponentialMovingAverage(0.9, self.val_step)
        self.val_accuracy = tf.get_variable('val_accuracy', [], dtype='float',
            initializer=tf.constant_initializer(0.0), trainable=False)
        self.val_accuracy_apply = tf.group(
            self.val_step.assign_add(1),
            ema.apply([self.val_accuracy]))
        val_accuracy_avg = ema.average(self.val_accuracy)
        tf.scalar_summary('val_accuracy_avg', val_accuracy_avg)


    def update_val_accuracy(self, accuracy):
        self.sess.run(self.val_accuracy_apply, { self.val_accuracy: accuracy })

    def _build_action(self, x, name, num_possible_actions, labels):
        return prob, loss

    def act(self, observation, is_training=True):
        #print "observation mean", np.mean(observation)
        #print "observation shape", observation.shape
        assert observation.shape == (FLAGS.glimpse_size, FLAGS.glimpse_size, 3)

        # If we are training, we randomly choose an action with probability
        # dqn_epsilon. dqn_epsilon starts at 1.0 and decays to 0.1 over
        # 1,000,000 steps. So at the beginning we only make random actions
        # while towards the end we determine the action 90% of the time.
        if is_training:
            step = self.sess.run(self.global_step)
            e = (-0.9 / 1000000) * step + 1.0
            dqn_epsilon = max(e, 0.1)

            #print step, "dqn_epsilon", dqn_epsilon

            # With probability dqn_epsilon select a random action.
            if random.random() < dqn_epsilon:
                action = np.random.randint(0, num_actions)
                return action

        # Otherwise select an action that maximizes Q
        assert self.current_ep.step < FLAGS.max_episode_steps

        action = self.sess.run(self.last_maximizing_action, {
           self.is_training: False,
           self.inputs: self.observations,
        })

        return action

    def reset(self, observation, observation_params):
        self.observations = observation[np.newaxis,:]
        self.current_ep = Episode(observation_params)

    def store(self, observation, action, reward, done, is_training, observation_params):
        ep = self.current_ep
        ep.store(observation_params, action, reward)

        done = (done or ep.step >= FLAGS.max_episode_steps)

        prev_num_observations = self.observations.shape[0]
        self.observations = np.concatenate((self.observations, observation[np.newaxis,:]))
        assert self.observations.shape[0] == 1 + prev_num_observations

        if done:
            ep.done()

            if DEBUG and is_training:
                # This is expensive!
                np.testing.assert_allclose(ep.obvs, self.observations)

            self.observations = None
            log("episode done. num_actions %d num_frames %d" % (ep.num_actions, ep.num_frames))

            if is_training:
                self.replay_memory.store(ep)

            self.current_ep = None

    def train(self):
        step = self.sess.run(self.global_step)
        write_summary = (step % 10 == 0 and step > 1)
        # Sample random minibatch of transititons

        if self.replay_memory.count() < 1: return

        random_episode = self.replay_memory.sample()

        assert random_episode._done

        # random_frame is "j" in the dqn paper. so it can be anything but the last.
        random_frame = np.random.randint(0, random_episode.num_frames - 1)

        #print "num frames", random_episode.num_frames
        #print "rewards", random_episode.rewards
        #print "actions", random_episode.actions
        #print "random_frame", random_frame

        is_terminal = (random_frame + 1 == random_episode.num_frames - 1)
        reward = random_episode.rewards[random_frame]
        action = random_episode.actions[random_frame]

        #print "is_terminal", is_terminal
        #print "reward", reward
        #print "action", action

        assert reward == 0 or is_terminal

        # phi_j and phi_j+1 
        obvs = random_episode.obvs[random_frame:random_frame+2]

        i = [self.train_op, self.loss]

        if write_summary:
            i.append(self.summary_op)

        o = self.sess.run(i, {
            self.is_training: True,
            self.is_terminal: is_terminal,
            self.inputs: obvs,
            self.reward: reward,
            self.action: action,
        })

        loss_value = o[1]

        if write_summary:
            summary_str = o[2]
            self.summary_writer.add_summary(summary_str, step)

        if step % 5 == 0:
            print "step %d: %f loss" % (step, loss_value)

        if step % 1000 == 0 and step > 0:
            print 'save checkpoint'
            self.saver.save(self.sess, self.checkpoint_path, global_step=self.global_step)

def main(_):
    print_flags()

    os.environ["IMAGENET_DIR"] = "/Users/ryan/data/imagenet-small/imagenet-small-train"
    env = gym.make('Attention%d-v0' % FLAGS.glimpse_size)

    os.environ["IMAGENET_DIR"] = "/Users/ryan/data/imagenet-small/imagenet-small-val"
    env_val = gym.make('Attention%d-v0' % FLAGS.glimpse_size)

    #env.monitor.start('/tmp/attention', force=True)

    sess = tf.Session(config=tf.ConfigProto(
                      allow_soft_placement=True,
                      log_device_placement=False))

    agent = Agent(sess)

    for i_episode in xrange(1, FLAGS.num_episodes):
        observation = env.reset()

        agent.reset(observation, env.backdoor_observation_params())

        for t in xrange(FLAGS.max_episode_steps):
            mode = 'human' if FLAGS.show_train_window else 'rgb_array'
            env.render(mode=mode)
            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)
            #print "action, reward, done", (action, reward, done)
            agent.store(
                observation=observation,
                observation_params=env.backdoor_observation_params(),
                action=action,
                reward=reward,
                done=done,
                is_training=True)
            agent.train()
            if done: break

        if i_episode % 10 == 0:
            validation(agent, env_val)

    #env.monitor.close()

def validation(agent, env_val):
    correct = 0
    total = 10
    for _ in xrange(0, total):
        observation = env_val.reset()
        agent.reset(observation, env_val.backdoor_observation_params())
        debug_str = ''
        for t in xrange(FLAGS.max_episode_steps):
            env_val.render(mode='rgb_array')
            action = agent.act(observation, is_training=False)

            debug_str += action_human_str(action) + ' '

            observation, reward, done, _ = env_val.step(action)
            agent.store(
                observation=observation,
                observation_params=env_val.backdoor_observation_params(),
                action=action,
                reward=reward,
                done=done,
                is_training=False)
            if reward > 0:
                assert done
                correct += 1
                debug_str += "GOOD"
            if done: break
        print debug_str

    accuracy = float(correct) / total
    agent.update_val_accuracy(accuracy)
    print "validation accuracy %.2f" % accuracy

def print_flags():
    flags = FLAGS.__dict__['__flags']
    for f in flags:
        print f, flags[f]

if __name__ == '__main__':
    tf.app.run()
