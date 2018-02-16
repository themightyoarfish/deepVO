import tensorflow as tf
from math import ceil

class VOModel(object):

    '''Model class of the RCNN for visual odometry.'''

    def __init__(self, image_shape, memory_size, sequence_length):
        '''
        Parameters
        ----------
        image_shape :   tuple
        '''

        with tf.name_scope('inputs'):
            h, w, c = image_shape
            # TODO: Resize images before stacking. Maybe do that outside of the graph?
            self.input_images = tf.placeholder(tf.uint8, shape=[None, sequence_length, h, w, c],
                                               name='imgs')
            self.target_poses = tf.placeholder(tf.float32, shape=[6], name='poses')
            self.batch_size   = tf.placeholder(tf.uint8, shape=[], name='batch_size')
            self.hidden_state = tf.placeholder(tf.float32, shape=(None, memory_size),
                                               name='hidden_state')
            self.cell_state   = tf.placeholder(tf.float32, shape=(None, memory_size),
                                               name='cell_state')
        with tf.name_scope('cnn'):
            self.build_cnn()

        with tf.name_scope('rnn'):
            self.build_rnn()

    def build_cnn(self, use_dropout=False):
        '''Create all the conv layers as specified in the paper.'''

        # biases initialise with a small constant
        bias_initializer = tf.constant_initializer(0.01)

        # kernels initialise according to He et al.
        def kernel_initalizer(k):
            return tf.random_normal_initializer(stddev=np.sqrt(2 / k))

        ksizes     = [7,  5,   5,   3,   3,   3,   3,   3,   3]
        strides    = [2,  2,   2,   1,   2,   1,   2,   1,   2]
        n_channels = [64, 128, 256, 256, 512, 512, 512, 512, 1024]

        next_layer_input = self.input_images
        for index, ksize, stride, channels in enumerate(zip(ksizes, strides, n_channels)):
            with tf.name_scope(f'conv{index}'):
                # no relu for last layer
                activation = tf.nn.relu if index < len(ksize) - 1 else None
                next_layer_input = tf.layers.conv2d(next_layer_input,
                                                    channels,
                                                    kernel_size=[ksize, ksize],
                                                    stride,
                                                    padding='SAME',
                                                    activation=activation,
                                                    kernel_initalizer=kernel_initalizer(ksize),
                                                    bias_initializer=bias_initializer)
        self.conv = next_layer_input

    def build_rnn(self):
        '''Create all recurrent layers as specified in the paper.'''
        pass

    @staticmethod
    def resize_to_multiple(images, multiples):
        '''Resize a batch of images in the height and width dimensions so their size are an integer
        multiple of some value.

        Parameters
        ----------
        images  :   tf.Tensor
                    Tensor of shape [batch, height, width, channels]
        multiples   :   int or tuple
                        The value/s that should evenly divide the resized image's dimensions
        '''
        _, h, w, _ = images.get_shape()
        # if only one multiple, assume it's the value to use for all dims
        if not isinstance(multiples, tuple):
            multiples = multiples * 2
        new_h, new_w = [int(ceil(input_shape[0] / multiples[0])),
                        int(ceil(input_shape[1] / multiples[1]))]
        return tf.image.resize_images(images, [new_h, new_w])
