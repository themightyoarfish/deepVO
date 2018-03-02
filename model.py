import tensorflow as tf
from math import ceil
from tensorflow.contrib.rnn import *
import numpy as np
from utils import tensor_from_lstm_tuple, OptimizerSpec, resize_to_multiple, conv_layer

flownet_prefix = 'FlowNetS'
flownet_kernel_suffix = 'weights'
flownet_bias_suffix = 'biases'
flownet_layer_names = [
    'conv1',
    'conv2',
    'conv3',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6',
    'conv6_1',
]


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

    def __init__(self, image_shape,
                 memory_size,
                 sequence_length,
                 batch_size,
                 optimizer_spec=None,
                 resize_images=False,
                 is_training=True,
                 use_flownet=False):
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
        if not optimizer_spec:
            optimizer_spec = OptimizerSpec(kind='Adagrad', learning_rate=0.001)
        optimizer = optimizer_spec.create()
        self.is_training = is_training
        self.batch_size = batch_size
        self.use_flownet = use_flownet
        ############################################################################################
        #                                          Inputs                                          #
        ############################################################################################
        with tf.variable_scope('inputs'):
            h, w, c = image_shape
            # TODO: Resize images before stacking. Maybe do that outside of the graph?
            self.input_images = tf.placeholder(tf.float32, shape=[batch_size, sequence_length, h, w, 2 * c],
                                               name='imgs')
            if resize_images:
                self.input_images = resize_to_multiple(self.images, 64)

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
        # TODO: Try different dropout schemes. On the small training set, every kind of dropout
        # prevents convergence
        n = len(ksizes)
        keep_probs = np.linspace(0.5, 1, num=n)
        lstm_keep_probs = [0.7, 0.8]

        self.cnn_activations = []
        # we call cnn() in a loop, but the variables will be reused after first creation
        for idx in range(sequence_length):
            stacked_image = self.input_images[:, idx, :]
            cnn_activation = self.cnn(stacked_image,
                                      ksizes,
                                      strides,
                                      n_channels,
                                      reuse=tf.AUTO_REUSE)
            if self.is_training:
                self.cnn_activations.append(tf.nn.dropout(cnn_activation,
                                                          keep_prob=keep_probs[idx]))
            else:
                self.cnn_activations.append(cnn_activation)

        # flatten cnn output for each batch element
        rnn_inputs = [tf.reshape(conv, [batch_size, -1])
                      for conv in self.cnn_activations]

        ############################################################################################
        #                                           LSTM                                           #
        ############################################################################################
        with tf.variable_scope('rnn'):
            '''Create all recurrent layers as specified in the paper.'''
            lstm0 = LSTMCell(memory_size, state_is_tuple=True)
            lstm1 = LSTMCell(memory_size, state_is_tuple=True)
            if self.is_training:
                lstm0 = tf.contrib.rnn.DropoutWrapper(lstm0, output_keep_prob=lstm_keep_probs[0])
                lstm1 = tf.contrib.rnn.DropoutWrapper(lstm1, output_keep_prob=lstm_keep_probs[1])
            rnn = MultiRNNCell([lstm0, lstm1])

            self.zero_state = rnn.zero_state(batch_size, tf.float32)

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

            rnn_outputs, rnn_state = static_rnn(rnn,
                                                rnn_inputs,
                                                dtype=tf.float32,
                                                initial_state=(state0, state1),
                                                sequence_length=[sequence_length] * batch_size)
            rnn_outputs = tf.reshape(tf.concat(rnn_outputs, 1),
                                     [batch_size, sequence_length, memory_size])
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
            y_t, y_r = tf.split(y, 2, axis=2)
            x_t, x_r = tf.split(self.target_poses, 2, axis=2)

        self.loss = self.loss_function((x_t, x_r), (y_t, y_r))
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
        '''
        error_t = tf.losses.mean_squared_error(targets[0], predictions[0], reduction=tf.losses.Reduction.SUM)
        error_r = tf.losses.mean_squared_error(targets[1], predictions[1], weights=rot_weight,
                                               reduction=tf.losses.Reduction.SUM)
        return (error_r + error_t) / self.batch_size

    def cnn(self, input, ksizes, strides, n_channels, use_dropout=False, reuse=True):
        '''Create all the conv layers as specified in the paper.'''

        assert len(ksizes) == len(strides) == len(n_channels), ('Kernel, stride and channel specs '
                                                                'must have same length')
        with tf.variable_scope(flownet_prefix if self.use_flownet else 'cnn', reuse=reuse):

            # biases initialise with a small constant
            bias_initializer = tf.constant_initializer(0.01)

            # kernels initialise according to He et al.
            def kernel_initializer(k):
                return tf.random_normal_initializer(stddev=np.sqrt(2 / k))

            output = input

            for index, [ksize, stride, channels] in enumerate(zip(ksizes, strides, n_channels)):
                with tf.variable_scope(flownet_layer_names[index] if self.use_flownet else f'conv{index}'):
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
                                            var_names=(kernel_name, bias_name))

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
        return session.run(self.loss, feed_dict={self.input_images: input_batch,
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

    def train(self, session, input_batch, pose_batch, initial_states=None):
        '''Train the network.

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
        if initial_states is None:
            initial_states = tensor_from_lstm_tuple(self.get_zero_state(session))

        return session.run([self.train_step, self.loss, self.rnn_state], feed_dict={self.input_images: input_batch,
                                                            self.target_poses: pose_batch,
                                                            self.lstm_states: initial_states})

    def load_flownet(self, session, filename):
        # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        # print_tensors_in_checkpoint_file(file_name=filename, tensor_name='', all_tensors=False)
        cnn_vars = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=flownet_prefix) if 'optimizer' not in var.name]
        # with tf.variable_scope('cnn', reuse=True):
        #     # this must be done in the same scope as was used to create the variables.
        #     cnn_vars = [tf.get_variable(name) for name in cnn_var_names]
        # assert len(cnn_vars) == len(flow_var_names)

        # var_map = dict(zip(flow_var_names, cnn_vars))
        restorer = tf.train.Saver(cnn_vars)
        restorer.restore(session, filename)
