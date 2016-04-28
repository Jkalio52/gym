import gym
import time
import tensorflow as tf
import numpy as np
import os
import sys

import resnet as resnet

from tensorflow.models.rnn.rnn_cell import BasicLSTMCell


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('continue', False, 'resume from latest saved state')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('num_episodes', 100000, 'number of epsidoes to run')
tf.app.flags.DEFINE_integer('glimpse_size', 96, '96 or 224')
tf.app.flags.DEFINE_integer('hidden_size', 1024, '')
tf.app.flags.DEFINE_integer('max_episode_steps', 10, '')
tf.app.flags.DEFINE_string('train_dir', '/tmp/agent_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('restore_resnet', '', 'path to resnet ckpt to restore from')


class Episode:
    observations = []
    actions = []
    rewards = []
    step = 0
    total_reward = 0

    def store(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_reward += reward
        self.step += 1

    def _labels(self, action_index, possible_actions):
        labels = np.zeros((self.step, possible_actions))
        for i in range(0, self.step):
            quit = self.actions[i][action_index]
            labels[i, int(quit)] = 1.0
        return labels

    def quit_labels(self):
        return self._labels(0, 2)

    def y_labels(self):
        return self._labels(2, 127)

    def x_labels(self):
        return self._labels(3, 127)

    def zoom_labels(self):
        return self._labels(4, 127)


def sample(probs):
    return np.random.choice(len(probs), 1, p=probs)[0]

class Agent(object):
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')

    def __init__(self, sess):
        self.sess = sess
        self.global_step = tf.get_variable('global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        self.observations = np.zeros((10, FLAGS.glimpse_size, FLAGS.glimpse_size, 3))

        self.episodes = [ Episode() ]

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
            resnet_variables_to_restore = {}
            # Correctly map the cnn/ prefix 
            for var in tf.get_collection(resnet.RESNET_VARIABLES):
                assert var.op.name.startswith('cnn/')
                n = var.op.name[len('cnn/'):] # remove prefix
                resnet_variables_to_restore[n] = var
            saver = tf.train.Saver(resnet_variables_to_restore)
            # '/Users/ryan/src/tensorflow-resnet/ResNet-L50.ckpt'
            saver.restore(self.sess, FLAGS.restore_resnet)
            print "done"

    def _build(self):
        inputs_shape = [None, FLAGS.glimpse_size, FLAGS.glimpse_size, 3]
        self.inputs = tf.placeholder('float', inputs_shape)

        self.is_training = tf.placeholder('bool', [], name='is_training')
        self.reward = tf.placeholder('float', [], name='reward')
        self.train_steps = tf.placeholder('int32', [], name='train_steps')
        self.quit_labels = tf.placeholder('float', [None, 2], name='quit_labels')
        self.classify_labels = tf.placeholder('float', [None, 1000], name='classify_labels')
        self.y_labels = tf.placeholder('float', [None, 127], name='y_labels')
        self.x_labels = tf.placeholder('float', [None, 127], name='x_labels')
        self.zoom_labels = tf.placeholder('float', [None, 127], name='zoom_labels')

        with tf.variable_scope('cnn'):
            # first axis of inputs is time. conflate with batch in cnn.
            # batch size is always 1 with this agent.
            x = self.inputs
            x = resnet.inference(x,
                                 is_training=self.is_training,
                                 num_blocks=[3, 4, 6, 3])
            # add batch dimension. output: batch=1, time, height, width, depth=3
            x = tf.expand_dims(x, 0)

        self.lstm = BasicLSTMCell(FLAGS.hidden_size)
        outputs, states = tf.nn.dynamic_rnn(self.lstm, x, dtype='float')

        assert outputs.get_shape().as_list() == [1,  None, self.lstm.output_size]
        outputs = tf.squeeze(outputs, squeeze_dims=[0]) # remove first axis

        assert states.get_shape().as_list() == [None, self.lstm.state_size]

        quit_prob, quit_loss = self._build_action(outputs, 'quit', 2, self.quit_labels)
        classify_prob, classify_loss = self._build_action(outputs, 'classify', 1000, self.classify_labels)
        y_prob, y_loss = self._build_action(outputs, 'y', 127, self.y_labels)
        x_prob, x_loss = self._build_action(outputs, 'x', 127, self.x_labels)
        zoom_prob, zoom_loss = self._build_action(outputs, 'zoom', 127, self.zoom_labels)

        self.probs = [ quit_prob, classify_prob, y_prob, x_prob, zoom_prob ]

        losses = [
          tf.reduce_mean(self.reward * quit_loss),
          tf.reduce_mean(classify_loss),
          tf.reduce_mean(self.reward * y_loss),
          tf.reduce_mean(self.reward * x_loss),
          tf.reduce_mean(self.reward * zoom_loss),
        ]

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        self.loss = tf.add_n(losses + regularization_losses, name='loss')

        # TODO need to use tf.GraphKeys.REGULARIZATION_LOSS
        tf.scalar_summary('loss', self.loss)


    def _build_action(self, x, name, num_possible_actions, labels):
        logits = tf.contrib.layers.fully_connected(x, num_possible_actions,
            weight_regularizer=tf.contrib.layers.l2_regularizer(0.00004), name=name)

        prob = tf.nn.softmax(logits)
        # labels is shaped [time, num_possible_actions]
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        return prob, loss

    @property
    def current_ep(self):
        return self.episodes[len(self.episodes)-1]

    def act(self, observation):
        i = self.current_ep.step
        assert i < FLAGS.max_episode_steps

        self.observations[i, :] = observation
        obvs = self.observations[:i+1,:]
        actions = self.sess.run(self.probs, {
           self.inputs: obvs,
           self.is_training: False,
        })

        batch_index = 0

        quit = sample(actions[0][batch_index])
        classify = sample(actions[1][batch_index])

        y_int = sample(actions[2][batch_index])
        y = 2 * (y_int / 127.0) - 1

        x_int = sample(actions[3][batch_index])
        x = 2 * (x_int / 127.0) - 1

        zoom_int = sample(actions[4][batch_index])
        zoom = zoom_int / 127.0

        return [quit, classify, y, x, zoom]


    def store(self, observation, action, reward, done, correct_answer):
        self.current_ep.store(observation, action, reward)

        if done or self.current_ep.step >= FLAGS.max_episode_steps:
            ep = self.episodes[-1]
            self.episodes.append(Episode())

            if done:
                self._train(ep, correct_answer)

    def _train(self, ep, correct_answer):
        # one hot vector
        classify_labels = np.zeros((ep.step, 1000))
        classify_labels[:, correct_answer] = 1

        feed_dict = {
          self.is_training: True,
          self.reward: ep.total_reward,
          self.train_steps: ep.step,
          self.inputs: self.observations[:ep.step,:],
          self.quit_labels: ep.quit_labels(),
          self.classify_labels: classify_labels,
          self.y_labels: ep.y_labels(),
          self.x_labels: ep.x_labels(),
          self.zoom_labels: ep.zoom_labels(),
        }

        step, _, loss_value = self.sess.run([self.global_step, self.train_op, self.loss], feed_dict)

        if step % 10 == 0:
            summary_str = self.sess.run(self.summary_op, feed_dict)
            self.summary_writer.add_summary(summary_str, step)

        print "Episode %d: %d steps, %f loss" % (step, ep.step, loss_value)

        if step % 1000 == 0:
            print 'save checkpoint'
            self.saver.save(self.sess, self.checkpoint_path, global_step=self.global_step)

def main(_):
    env = gym.make('Attention%d-v0' % FLAGS.glimpse_size)
    sess = tf.Session(config=tf.ConfigProto(
                      allow_soft_placement=True,
                      log_device_placement=False))

    agent = Agent(sess)

    sess.run(tf.initialize_all_variables())

    for i_episode in xrange(1, FLAGS.num_episodes):
        observation = env.reset()

        for t in xrange(FLAGS.max_episode_steps):
            env.render()
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            agent.store(observation, action, reward, done, info)
            if done: break

if __name__ == '__main__':
    tf.app.run()
