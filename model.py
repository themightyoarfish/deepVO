'''
.. module:: model

Contains :py:class:`VOModel`, an implementation of DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent
Convolutional Neural Networks by Wang et al.

.. moduleauthor:: Rasmus Diederichsen

'''
import tensorflow as tf
from math import ceil
from tensorflow.contrib.rnn import *
import numpy as np
from utils import tensor_from_lstm_tuple, OptimizerSpec, resize_to_multiple, conv_layer
from flownet import *


class VOModel(object):
    '''Model class of the RCNN for visual odometry.

    Attributes
    ----------
    input_images    :   tf.Placeholder
                        Float placeholder with shape ``(batch_size, sequence_length, h, w, c * 2)``.
                        This tensor contains the stacked input images.
    target_poses    :   tf.Placeholder
                        Float placeholder of shape ``(batch_size, sequence_length, 6)`` with 3
                        translational and 3 rotational components
    lstm_states :   tf.Placeholder
                    Float placeholder used to feed the initial lstm state into the network. The shape
                    is ``(2, 2, batch_size, memory_size)`` since we have 2 lstm cells and cell and hidden
                    states are contained in this tensor. THE CELL STATE (TUPLE MEMBER H) MUST COME
                    BEFORE THE HIDDEN STATE (TUPLE MEMBER C).
    sequence_length :   int
                        Length of the input sequences

    cnn_activations :   list(tf.Tensor)
                        List of cnn activations for all time steps
    zero_state  :   tuple(LSTMStateTuple)
                    Tuple of :py:class:`LSTMStateTuple` s for each lstm layer. Filled with zeros.
    rnn_outputs :   tf.Tensor
                    outputs of the RNN
    rnn_state   :   tuple(LSTMStateTuple)
                    Final states of the lstms

    '''

    def __init__(self,
                 image_shape,
                 memory_size,
                 sequence_length,
                 optimizer_spec=None,
                 resize_images=False,
                 use_dropout=True,
                 use_flownet=False):
        '''
        Parameters
        ----------
        image_shape :   tuple
        memory_size :   int
                        LSTM state size (identical for both layers)
        sequence_length :   int
                            Length of the video stream
        optimizer_spec  :   OptimizerSpec
                            Specification of the optimizer
        resize_images   :   bool
                            Rezise images to a multiple of 64
        use_dropout :   bool
                        Do not use dropout for LSTM cells
        use_flownet :   bool
                        Name CNN vars according to flownet naming scheme. You *must* call
                        :py:meth:`load_flownet`  before pushing stuff through the graph.
        '''
        if not optimizer_spec:
            optimizer_spec = OptimizerSpec(kind='Adagrad', learning_rate=0.001)
        optimizer = optimizer_spec.create()
        self.use_dropout = use_dropout
        self.use_flownet = use_flownet
        self.sequence_length = sequence_length
        ############################################################################################
        #                                          Inputs                                          #
        ############################################################################################
        with tf.variable_scope('inputs'):
            h, w, c = image_shape
            self.input_images = tf.placeholder(tf.float32, shape=[None, sequence_length, h, w, 2 * c],
                                               name='imgs')
            if resize_images:
                self.input_images = resize_to_multiple(self.images, 64)

            self.target_poses = tf.placeholder(tf.float32, shape=[None, sequence_length, 6],
                                               name='target_poses')
            # this placeholder is used for feeding both the cell and hidden states of both lstm
            # cells. The cell state comes before the hidden state
            N_lstm = 2
            self.lstm_states = tf.placeholder(tf.float32, shape=(N_lstm, 2, None, memory_size),
                                              name='LSTM_states')
            self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

        ############################################################################################
        #                                       Convolutions                                       #
        ############################################################################################
        ksizes     = [7,  5,   5,   3,   3,   3,   3,   3,   3]
        strides    = [2,  2,   2,   1,   2,   1,   2,   1,   2]
        n_channels = [64, 128, 256, 256, 512, 512, 512, 512, 1024]

        self.cnn_activations = []
        # we call cnn() in a loop, but the variables will be reused after first creation
        for idx in range(sequence_length):
            stacked_image = self.input_images[:, idx, :]
            cnn_activation = self.cnn(stacked_image,
                                      ksizes,
                                      strides,
                                      n_channels,
                                      reuse=tf.AUTO_REUSE)
            self.cnn_activations.append(cnn_activation)

        # compute number of activations for flattening the conv output
        def num_activations(conv):
            return np.prod(conv.shape[1:].as_list())

        # flatten cnn output for each batch element
        rnn_inputs = [tf.reshape(conv, [self.batch_size, num_activations(conv)])
                      for conv in self.cnn_activations]

        ############################################################################################
        #                                           LSTM                                           #
        ############################################################################################
        with tf.variable_scope('rnn'):
            '''Create all recurrent layers as specified in the paper.'''
            lstm0 = LSTMCell(memory_size, state_is_tuple=True)
            lstm1 = LSTMCell(memory_size, state_is_tuple=True)
            if self.use_dropout:
                lstm_keep_probs = [0.7, 0.8]
                lstm0 = tf.contrib.rnn.DropoutWrapper(lstm0, output_keep_prob=lstm_keep_probs[0])
                lstm1 = tf.contrib.rnn.DropoutWrapper(lstm1, output_keep_prob=lstm_keep_probs[1])
            self.rnn = MultiRNNCell([lstm0, lstm1])
            self.zero_state = self.rnn.zero_state(self.batch_size, tf.float32)

            # first decompose state input into the two layers
            states0 = self.lstm_states[0, ...]
            states0 = self.lstm_states[1, ...]

            # then retrieve two memory_size-sized tensors from each state item
            states0_list  = tf.unstack(states0, num=2)
            cell_state0   = states0_list[0]
            hidden_state0 = states0_list[1]

            states1_list  = tf.unstack(states0, num=2)
            cell_state1   = states1_list[0]
            hidden_state1 = states1_list[1]

            # finally, create the state tuples
            state0 = LSTMStateTuple(c=hidden_state0, h=cell_state0)
            state1 = LSTMStateTuple(c=hidden_state1, h=cell_state1)

            sequence_lengths = tf.ones((self.batch_size,), dtype=tf.int32) * sequence_length
            rnn_outputs, rnn_state = static_rnn(self.rnn,
                                                rnn_inputs,
                                                dtype=tf.float32,
                                                initial_state=(state0, state1),
                                                sequence_length=sequence_lengths)
            rnn_outputs = tf.reshape(tf.concat(rnn_outputs, 1),
                                     [self.batch_size, sequence_length, memory_size])
            self.rnn_state = tensor_from_lstm_tuple(rnn_state)

        ############################################################################################
        #                                       Output layer                                       #
        ############################################################################################
        with tf.variable_scope('feedforward'):
            n_rnn_output       = memory_size  # number of activations per batch
            kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2 / n_rnn_output))
            # predictions
            y = tf.layers.dense(rnn_outputs, 6, kernel_initializer=kernel_initializer)
            # decompose into translational and rotational component
            self.y_t, self.y_r = tf.split(y, 2, axis=2)
            self.x_t, self.x_r = tf.split(self.target_poses, 2, axis=2)
            self.predictions = (self.y_t, self.y_r)

        self.loss = self.loss_function((self.x_t, self.x_r), (self.y_t, self.y_r))
        with tf.variable_scope('optimizer'):
            self.train_step = optimizer.minimize(self.loss)

    def loss_function(self, targets, predictions, rot_weight=100):
        '''Create MSE loss.

        Parameters
        ----------
        targets :   tuple
                    Tuple of two (batch_size, sequence_length, 3)-sized tensors, where the last
                    dimension of the first element is the translation and the last dimension of the
                    second element is the three euler angles
        predictions :   tuple
                    Tuple of two (batch_size, sequence_length, 3)-sized tensors, where the last
                    dimension of the first element is the translation and the last dimension of the
                    second element is the three euler angles
        rot_weight  :   float
                        Weight to scale the rotational error with. See paper equation (5)

        Returns
        -------
        tf.Tensor
            Scalar float tensor
        '''
        error_t = tf.losses.mean_squared_error(targets[0], predictions[0], reduction=tf.losses.Reduction.MEAN)
        # from
        # https://stackoverflow.com/questions/46355068/keras-loss-function-for-360-degree-prediction
        diff_r = targets[1] - predictions[1]
        angle_differences = tf.atan2(tf.sin(diff_r), tf.cos(diff_r))
        error_r = tf.reduce_sum(tf.square(angle_differences)) * rot_weight / tf.cast(self.batch_size, tf.float32)
        return error_r + error_t

    def cnn(self, input, ksizes, strides, n_channels, use_dropout=False, reuse=True):
        '''Create all the conv layers as specified in the paper.'''

        assert len(ksizes) == len(strides) == len(n_channels), ('Kernel, stride and channel specs '
                                                                'must have same length')
        outer_scope_name = flownet_prefix if self.use_flownet else 'cnn'
        with tf.variable_scope(outer_scope_name, reuse=reuse):

            # biases initialise with a small constant
            bias_initializer = tf.constant_initializer(0.01)

            # kernels initialise according to He et al.
            def kernel_initializer(k):
                return tf.random_normal_initializer(stddev=np.sqrt(2 / k))

            output = input

            for index, [ksize, stride, channels] in enumerate(zip(ksizes, strides, n_channels)):
                inner_scope_name = flownet_layer_names[index] if self.use_flownet else f'conv{index}'
                with tf.variable_scope(inner_scope_name):
                    # no relu for last layer
                    activation = tf.nn.relu if index < len(ksizes) - 1 else None

                    if not self.use_flownet:
                        output = tf.layers.conv2d(output,
                                                  channels,
                                                  kernel_size=[ksize, ksize],
                                                  strides=stride,
                                                  padding='SAME',
                                                  activation=activation,
                                                  kernel_initializer=kernel_initializer(ksize),
                                                  bias_initializer=bias_initializer)
                    else:
                        # since we need control over the variable namings, we cannot use
                        # tf.layers.conv2d
                        bias_name = flownet_bias_suffix
                        kernel_name = flownet_kernel_suffix
                        output = conv_layer(output,
                                            channels,
                                            kernel_width=ksize,
                                            strides=stride,
                                            activation=activation,
                                            kernel_initializer=kernel_initializer(ksize),
                                            bias_initializer=bias_initializer,
                                            use_bias=True,
                                            padding='SAME',
                                            var_names=(kernel_name, bias_name),
                                            trainable=False)

            return output

    def get_zero_state(self, session, batch_size):
        '''Obtain the RNN zero state.

        Parameters
        ----------
        session :   tf.Session
                    Session to execute op in

        Returns
        -------
        LSTMStateTuple
            RNN zero state
        '''

        return session.run(self.zero_state, feed_dict={self.batch_size: batch_size})

    def get_rnn_output(self, session, input_batch, pose_batch, initial_states=None):
        '''Run some input through the cnn net, followed by the rnn net

        Parameters
        ----------
        session :   tf.Session
                    Session to execute op in
        input_batch  :  np.ndarray
                        Array of shape ``(batch_size, sequence_length, h, w, 6)`` where two consecutive
                        rgb images are stacked together.
        pose_batch  :   np.ndarray
                        Array of shape ``(batch_size, sequence_length, 6)`` with Poses
        initial_states   :   np.ndarray
                            Array of shape ``(2, 2, batch_size, memory_size)``

        Returns
        -------
        tuple(np.ndarray)
            Output of ``rnn_state`` operation (list of time steps)
        '''
        batch_size = input_batch.shape[0]
        return session.run(self.rnn_state, feed_dict={self.batch_size: batch_size,
                                                      self.input_images: input_batch,
                                                      self.target_poses: pose_batch,
                                                      self.lstm_states: initial_states})

    def get_cnn_output(self, session, input_batch, pose_batch):
        '''Run some input through the cnn net.

        Parameters
        ----------
        session :   tf.Session
                    Session to execute op in
        input_batch  :  np.ndarray
                        Array of shape ``(batch_size, sequence_length, h, w, 6)`` where two consecutive
                        rgb images are stacked together.
        pose_batch :   np.ndarray
                        Array of shape ``(batch_size, sequence_length, 6)`` with Poses

        Returns
        -------
        tuple(np.ndarray)
            Outputs of ``cnn_activation`` for each time step
        '''
        batch_size = input_batch.shape[0]
        return session.run(self.cnn_activations, feed_dict={self.batch_size: batch_size,
                                                            self.input_images: input_batch,
                                                            self.target_poses: pose_batch})

    def train(self, session, input_batch, pose_batch, initial_states=None, return_prediction=False):
        '''Train the network.

        Parameters
        ----------
        session :   tf.Session
                    Session to execute op in
        input_batch  :  np.ndarray
                        Array of shape ``(batch_size, sequence_length, h, w, 6)`` where two consecutive
                        rgb images are stacked together.
        pose_batch  :   np.ndarray
                        Array of shape ``(batch_size, sequence_length, 6)`` with Poses
        initial_states   :  np.ndarray
                            Array of shape ``(2, 2, batch_size, memory_size)``

        Returns
        -------
        tuple(np.ndarray)
            Outputs of the ``train_step``, ``loss``, and ``rnn_state`` operations, and optionally
            the predictions for r and t at the front
        '''
        batch_size = input_batch.shape[0]

        if initial_states is None:
            zero_state = self.get_zero_state(session, batch_size)
            initial_states = tensor_from_lstm_tuple(zero_state)

        if return_prediction:
            fetches = [self.y_t, self.y_r,
                       self.train_step, self.loss, self.rnn_state]
        else:
            fetches = [self.train_step, self.loss, self.rnn_state]

        return session.run(fetches,
                           feed_dict={self.batch_size: batch_size,
                                      self.input_images: input_batch,
                                      self.target_poses: pose_batch,
                                      self.lstm_states: initial_states})

    def load_flownet(self, session, filename):
        '''Load flownet weights into the conv net.

        Parameters
        ----------
        session :   tf.Session
                    Session to restore weights in
        filename    :   str
                        Base name of the checkpoints file (e.g ``"flownet-S.ckpt-0"`` if you have
                        ``"flownet-S.ckpt-0.data-00000-of-00001"``, ``"flownet-S.ckpt-0.meta"``, and
                        ``"flownet-S.ckpt-0.index"``)
        '''
        cnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=flownet_prefix)
        restorer = tf.train.Saver(cnn_vars)
        restorer.restore(session, filename)

    def test(self, session, input_batch, pose_batch, initial_states=None):
        '''Get network predictions for some input sequence and compute average error.

        Parameters
        ----------
        session :   tf.Session
                    Session to run ops in
        input_batch  :  np.ndarray
                        Array of shape ``(batch_size, sequence_length, h, w, 6)`` where two consecutive
                        rgb images are stacked together.
        pose_batch  :   np.ndarray
                        Array of shape ``(batch_size, sequence_length, 6)`` with Poses
        '''
        batch_size = input_batch.shape[0]

        if initial_states is None:
            initial_states = tensor_from_lstm_tuple(self.get_zero_state(session, batch_size))

        fetches = [*self.predictions, self.loss, self.rnn_state]
        y_t, y_r, loss, states = session.run(fetches,
                                             feed_dict={self.batch_size: batch_size,
                                                        self.target_poses: pose_batch,
                                                        self.input_images: input_batch,
                                                        self.lstm_states: initial_states})
        return y_t, y_r, loss, states
