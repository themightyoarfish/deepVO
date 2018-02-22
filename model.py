import tensorflow as tf
from math import ceil
from tensorflow.contrib.rnn import *
import numpy as np
from utils import array_from_lstm_tuple


class VOModel(object):

    '''Model class of the RCNN for visual odometry.

    Attributes
    ----------
    input_images    :   tf.Placeholder
                        Float placeholder with shape (batch_size, sequence_length, h, w, c * 2).
                        This tensor contains the stacked input images.
    target_poses    :   tf.Placeholder
                        Float placeholder of shape (batch_size, sequence_length, 6) with 3
                        translational and 3 rotational components
    lstm_states :   tf.Placeholder
                    Float placeholder used to feed the initial lstm state into the network. The shape
                    is (2, 2, batch_size, memory_size) since we have 2 lstm cells and cell and hidden
                    states are contained in this tensor. THE CELL STATE (TUPLE MEMBER H) MUST COME
                    BEFORE THE HIDDEN STATE (TUPLE MEMBER C).
    sequence_length :   int
                        Length of the input sequences (cannot be altered)

    cnn_activations :   list(tf.Tensor)
                        List of cnn activations for all time steps
    zero_state  :   tuple(LSTMStateTuple)
                    Tuple of LSTMStateTuples for each lstm layer. Filled with zeros.
    rnn_outputs :   tf.Tensor
                    outputs of the RNN
    rnn_state   :   tuple(LSTMStateTuple)
                    Final states of the lstms

    '''

    def __init__(self, image_shape, memory_size, sequence_length, batch_size):
        '''
        Parameters
        ----------
        image_shape :   tuple
        memory_size :   int
                        LSTM state size
        sequence_length :   int
                            Length of the video stream
        batch_size  :   int
                        Size of the batches for training (necessary for RNN state)
        '''

        ############################################################################################
        #                                          Inputs                                          #
        ############################################################################################
        with tf.variable_scope('inputs'):
            h, w, c = image_shape
            # TODO: Resize images before stacking. Maybe do that outside of the graph?
            self.input_images = tf.placeholder(tf.float32, shape=[batch_size, sequence_length, h, w, 2 * c],
                                               name='imgs')


            self.target_poses = tf.placeholder(tf.float32, shape=[batch_size, sequence_length, 6],
                                               name='poses')
            # this placeholder is used for feeding both the cell and hidden states of both lstm
            # cells. The cell state comes before the hidden state
            N_lstm = 2
            self.lstm_states = tf.placeholder(tf.float32, shape=(N_lstm, 2,  batch_size, memory_size),
                                               name='LSTM_states')
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

        # flatten cnn output for each batch element
        rnn_inputs = [tf.reshape(conv, [batch_size, -1])
                      for conv in self.cnn_activations]

        ############################################################################################
        #                                           LSTM                                           #
        ############################################################################################
        with tf.variable_scope('rnn'):
            '''Create all recurrent layers as specified in the paper.'''
            lstm1 = LSTMCell(memory_size, state_is_tuple=True)
            lstm2 = LSTMCell(memory_size, state_is_tuple=True)
            rnn   = MultiRNNCell([lstm1, lstm2])

            self.zero_state = rnn.zero_state(batch_size, tf.float32)

            # first decompose state input into the two layers
            states1 = self.lstm_states[0, ...]
            states2 = self.lstm_states[1, ...]

            # then retrieve two memory_size-sized tensors from each state item
            states1_list  = tf.unstack(states1, num=2)
            cell_state1   = states1_list[0]
            hidden_state1 = states1_list[1]

            states2_list  = tf.unstack(states2, num=2)
            cell_state2   = states2_list[0]
            hidden_state2 = states2_list[1]

            # finally, create the state tuples
            state1 = LSTMStateTuple(c=hidden_state1, h=cell_state1)
            state2 = LSTMStateTuple(c=hidden_state2, h=cell_state2)

            rnn_outputs, self.rnn_state = static_rnn(rnn,
                                                     rnn_inputs,
                                                     dtype=tf.float32,
                                                     initial_state=(state1, state2),
                                                     sequence_length=[sequence_length] * batch_size)
            rnn_outputs = tf.reshape(tf.concat(rnn_outputs, 1),
                                     [batch_size, sequence_length * memory_size])

        ############################################################################################
        #                                       Output layer                                       #
        ############################################################################################
        with tf.variable_scope('feedforward'):
            n_rnn_output = rnn_outputs.get_shape()[-1]  # number of activations per batch
            network_function = tf.layers.dense(rnn_outputs, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2 / n_rnn_output)))

        self.loss = loss(network_function, target_poses)

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
                                              strides=stride,
                                              padding='SAME',
                                              activation=activation,
                                              kernel_initializer=kernel_initializer(ksize),
                                              bias_initializer=bias_initializer,
                                              reuse=reuse   # TODO: test if needed if set in parent scope
                                              )

            return output

    def get_zero_state(self, session):
        '''Obtain the RNN zero state.

        Parameters
        ----------
        session :   tf.Session
                    Session to execute op in
        '''
        return session.run(self.zero_state)

    def get_rnn_output(self, session, input_batch, pose_batch, initial_states=None):
        '''Run some input through the cnn net, followed by the rnn net

        Parameters
        ----------
        session :   tf.Session
                    Session to execute op in
        input_batch  :  np.ndarray
                        Array of shape (batch_size, sequence_length, h, w, 6) where two consecutive
                        rgb images are stacked together.
        pose_batch  :   np.ndarray
                        Array of shape (batch_size, sequence_length, 6) with Poses
        initial_states   :   np.ndarray
                            Array of shape (2, 2, batch_size, memory_size)
        '''
        batch_size = input_batch.shape[0]

        if initial_states is None:
            initial_states = array_from_lstm_tuple(self.get_zero_state(session))

        return session.run(self.cnn_activations, feed_dict={self.input_images: input_batch,
                                                            self.target_poses: pose_batch,
                                                            self.lstm_states: initial_states})

    def get_cnn_output(self, session, input_batch, pose_batch):
        '''Run some input through the cnn net.

        Parameters
        ----------
        session :   tf.Session
                    Session to execute op in
        input_batch  :  np.ndarray
                        Array of shape (batch_size, sequence_length, h, w, 6) where two consecutive
                        rgb images are stacked together.
        pose_batch :   np.ndarray
                        Array of shape (batch_size, sequence_length, 6) with Poses
        '''
        return session.run(self.cnn_activations, feed_dict={self.input_images: input_batch,
                                                            self.target_poses: pose_batch})
