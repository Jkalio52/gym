import sys
import os
import gym
import time
import skimage.io # for some reason this needs to be loaded before tf
import tensorflow as tf
import numpy as np

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

    def labels(self):
        # This is horrible code. Please fix.
        quit_labels = np.zeros((self.step, 2))
        y_labels = np.zeros((self.step, 127))
        x_labels = np.zeros((self.step, 127))
        zoom_labels = np.zeros((self.step, 127))
        for i in range(0, self.step):
            a = self.actions[i]
            quit = int(a[0])
            y = int(a[2][0])
            x = int(a[2][1])
            zoom = int(a[2][2])
            quit_labels[i, quit] = 1.0
            y_labels[i, y] = 1.0
            x_labels[i, x] = 1.0
            zoom_labels[i, zoom] = 1.0
        return quit_labels, y_labels, x_labels, zoom_labels

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
            resnet_variables_to_restore = tf.get_collection(resnet.RESNET_VARIABLES)
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

        # CNN
        # first axis of inputs is time. conflate with batch in cnn.
        # batch size is always 1 with this agent.
        x = self.inputs
        x = resnet.inference(x,
                             is_training=self.is_training,
                             preprocess=True,
                             num_blocks=[3, 4, 6, 3])
        # add batch dimension. output: batch=1, time, height, width, depth=3
        x = tf.expand_dims(x, 0)
        # END CNN

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

        #print "observation mean", np.mean(observation)
        #print "observation shape", observation.shape

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

        attention = np.asarray([y, x, zoom])
        return (quit, classify, attention)


    def store(self, observation, action, reward, done, correct_answer):
        self.current_ep.store(observation, action, reward)

        if done or self.current_ep.step >= FLAGS.max_episode_steps:
            ep = self.episodes[-1]

            if done:
                self._train(ep, correct_answer, len(self.episodes))

            self.episodes.append(Episode())

    def _train(self, ep, correct_answer, step):
        # one hot vector
        classify_labels = np.zeros((ep.step, 1000))
        classify_labels[:, correct_answer] = 1

        quit_labels, y_labels, x_labels, zoom_labels = ep.labels()
        feed_dict = {
          self.is_training: True,
          self.reward: ep.total_reward,
          self.train_steps: ep.step,
          self.inputs: self.observations[:ep.step,:],
          self.quit_labels: quit_labels,
          self.classify_labels: classify_labels,
          self.y_labels: y_labels,
          self.x_labels: x_labels,
          self.zoom_labels: zoom_labels,
        }

        i = [self.train_op, self.loss]
        write_summary = (step % 10 == 0)
        if write_summary:
            i.append(self.summary_op)

        o = self.sess.run(i, feed_dict)

        loss_value = o[1]

        if write_summary:
            summary_str = o[2]
            self.summary_writer.add_summary(summary_str, step)

        print "Episode %d: %d steps, %f loss" % (step, ep.step, loss_value)

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

        for t in xrange(FLAGS.max_episode_steps):
            env.render()
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            agent.store(observation, action, reward, done, info)
            if done: break

    #env.monitor.close()

if __name__ == '__main__':
    tf.app.run()
