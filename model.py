import tensorflow as tf

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
            self.input_images = tf.placeholder(tf.uint8, shape=[None, sequence_length, h, w, 2 * c],
                                               name='imgs')
            self.target_poses = tf.placeholder(tf.float32, shape=[6], name='poses')
            self.batch_size   = tf.placeholder(tf.uint8, shape=[], name='batch_size')
            self.hidden_state = tf.placeholder(tf.float32, shape=(None, memory_size),
                                               name='hidden_state')
            self.cell_state   = tf.placeholder(tf.float32, shape=(None, memory_size),
                                               name='cell_state')
        with tf.name_scope('cnn'):
            self.build_cnn()

    def build_cnn(self):


