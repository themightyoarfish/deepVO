import tensorflow as tf
from math import ceil
from tensorflow.contrib.rnn import *
import numpy as np

class VOModel(object):

    '''Model class of the RCNN for visual odometry.'''

    def __init__(self, image_shape, memory_size, sequence_length):
        '''
        Parameters
        ----------
        image_shape :   tuple
        memory_size :   int
                        LSTM state size
        sequence_length :   int
                            Length of the video stream
        '''

        ############################################################################################
        #                                          Inputs                                          #
        ############################################################################################
        with tf.variable_scope('inputs'):
            h, w, c = image_shape
            # TODO: Resize images before stacking. Maybe do that outside of the graph?
            self.input_images = tf.placeholder(tf.float32, shape=[None, sequence_length, h, w, 2 * c],
                                               name='imgs')

            self.target_poses = tf.placeholder(tf.float32, shape=[None, 6],
                                               name='poses')
            self.batch_size   = tf.placeholder(tf.int32, shape=[], name='batch_size')
            # self.hidden_state = tf.placeholder(tf.float32, shape=(None, memory_size),
            #                                    name='hidden_state')
            # self.cell_state   = tf.placeholder(tf.float32, shape=(None, memory_size),
            #                                    name='cell_state')
            self.sequence_length = sequence_length

        ############################################################################################
        #                                       Convolutions                                       #
        ############################################################################################
        ksizes     = [7,  5,   5,   3,   3,   3,   3,   3,   3]
        strides    = [2,  2,   2,   1,   2,   1,   2,   1,   2]
        n_channels = [64, 128, 256, 256, 512, 512, 512, 512, 1024]

        self.cnn_activations = []
        for idx in range(sequence_length):
            stacked_image = self.input_images[:, idx, :]
            self.cnn_activations.append(self.cnn(stacked_image,
                                            ksizes,
                                            strides,
                                            n_channels,
                                            reuse=tf.AUTO_REUSE))
            # with tf.conrrol_dependencies(self.cnn_activations):
            #     self.dummy_op = tf.identity(self.input_images)

        ############################################################################################
        #                                           LSTM                                           #
        ############################################################################################
        # with tf.variable_scope('rnn'):
        #     '''Create all recurrent layers as specified in the paper.'''
        #     lstm1 = LSTMCell(memory_size)
        #     lstm2 = LSTMCell(memory_size)
        #     rnn   = MultiRNNCell([lstm1, lstm2])

        #     self.zero_state = rnn.zero_state(self.batch_size, tf.float32)
        #     state           = LSTMStateTuple(c=self.hidden_state, h=self.cell_state)

        #     b, h, w,c = self.conv.get_shape()
        #     conv_outputs = tf.reshape(self.conv, [b, self.sequence_length, h * w * c])
        #     outputs, state  = static_rnn(rnn, conv_outputs, dtype=tf.float32, initial_state=state)


    def cnn(self, input, ksizes, strides, n_channels, use_dropout=False, reuse=True):
        '''Create all the conv layers as specified in the paper.'''

        assert len(ksizes) == len(strides) == len(n_channels), ('Kernel, stride and channel specs '
                                                                'must have same length')
        with tf.variable_scope('cnn', reuse=True):

            # biases initialise with a small constant
            bias_initializer = tf.constant_initializer(0.01)

            # kernels initialise according to He et al.
            def kernel_initializer(k):
                return tf.random_normal_initializer(stddev=np.sqrt(2 / k))

            output = input

            for index, [ksize, stride, channels] in enumerate(zip(ksizes, strides, n_channels)):
                with tf.variable_scope(f'conv{index}'):
                    # no relu for last layer
                    activation = tf.nn.relu if index < len(ksizes) - 1 else None

                    output = tf.layers.conv2d(output,
                                              channels,
                                              kernel_size=[ksize, ksize],
                                              strides=[stride, stride],
                                              padding='SAME',
                                              activation=activation,
                                              kernel_initializer=kernel_initializer(ksize),
                                              bias_initializer=bias_initializer,
                                              reuse=reuse   # TODO: test if needed if set in parent scope
                                              )

            return output

    def get_zero_state(self, session, batch_size):
        '''Obtain the RNN zero state.

        Parameters
        ----------
        session :   tf.Session
        Session to execute op in
        batch_size  :   int
        Batch size (influences the size of the RNN state)
        '''
        return session.run(self.zero_state, feed_dict={self.batch_size: batch_size})

    def get_cnn_output(self, session, input_batch, label_batch, initial_state=None):
        '''Run some input through the cnn net.

        Parameters
        ----------
        session :   tf.Session
        Session to execute op in
        input_batch  :  np.ndarray
        Array of shape (batch_size, sequence_length, h, w, 6) were two consecutive
        rgb images are stacked together.
        label_batch :   np.ndarray
        Array of shape (batch_size, sequence_length, 7) with Poses
        initial_state   :   LSTMStateTuple (aka namedtuple(c,h))
        Previous state
        '''
        batch_size = input_batch.shape[0]

        # if initial_state is None:
        #     initial_state = self.get_zero_state(session, batch_size)

        return session.run(self.cnn_activations, feed_dict={self.input_images: input_batch,
                                                            self.target_poses: label_batch,
                                                            self.batch_size: batch_size})
