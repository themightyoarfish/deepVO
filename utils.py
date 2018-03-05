'''
.. module:: utils
    Miscellaneous functions for data processing and batching. This module defines - among other things -
    :py:class:`OptimizerSpec` for specifying optimizers, and :py:class:`DataManager` to partition
    the data into batches.

.. moduleauthor Rasmus Diederichsen, Alexander Mock
'''
import numpy as np


def tensor_from_lstm_tuple(tuples, validate_shape=False):
    '''Create a tensor from a tuple of :py:class:`tf.contrib.rnn.LSTMStateTuple` s.

    .. note:: Error checks
        We do not check all possible error cases. For instance, the different LSTMStateTuples could
        not only have differing shapes (which we check for to some extend see ``validate_shape``
        parameter), but also the state members ``c`` and ``h`` could differ in their data type (Tensor,
        array), which we _do not_ check.


    Parameters
    ----------
    tuples  :   tuple(LSTMStateTuple)
                Tuple of N_lstm ``LSTMStateTuple`` s where each of the tuples has members of shape
                ``(batch_size, memory_size)``
    validate_shape  :   bool
                        Enforce identical shapes of all cell and memory states. This entails that
                        all dimensions must be known. When using variable batch size, set to
                        ``False`` and ensure the shapes are identical at runtime.

    Returns
    -------
    tf.Tensor or np.ndarray
        Tensor of shape ``(N_lstm, 2, batch_size, memory_size)`` with cell and hidden states per
        lstm cell stacked together. An array is returned instead in case the LSTMStateTuple members
        are already fully-defined arrays
    '''
    import tensorflow as tf
    # one state tuple has two members of shape (batch_size, memory_size)
    N_lstm      = len(tuples)
    batch_size  = tuples[0].c.shape[0]
    memory_size = tuples[0].c.shape[1]
    # return value. Since we don't know the dimensions upfront, make it a list instead of an array
    list_array  = [[None, None]] * N_lstm
    # explanation: see at return
    states_are_tensors = False

    for lstm_idx in range(N_lstm):
        lstm_state = tuples[lstm_idx]

        # check for incompatible shapes
        if validate_shape:
            # all dims must match
            if not ((batch_size, memory_size) == lstm_state.c.shape == lstm_state.h.shape):
                raise ValueError('All states must have the same dimenstion.')
        else:
            # only the memory_size must match, batch_size is assumed to match, but cannot be
            # verified
            if not (memory_size == lstm_state.c.shape[1] == lstm_state.h.shape[1]):
                raise ValueError('All states must have the same memory size.')

        if isinstance(lstm_state.c, tf.Tensor):
            states_are_tensors = True

        list_array[lstm_idx][0] = lstm_state.h  # cell state
        list_array[lstm_idx][1] = lstm_state.c  # hidden state

    ################################################################################################
    #  Why this? convert_to_tensor works when the list elements are tensors, but not if they are   #
    #  numpy arrays. This is probably a bug/missing feature. For this case, we must first convert  #
    #  the fully defined list of arrays to an array.                                               #
    #################################################################################################
    ######################################################################
    #  UPDATE: We now return an array if the states are already arrays.  #
    ######################################################################
    if not states_are_tensors:
        return np.array(list_array)
    else:
        return tf.convert_to_tensor(list_array)


# q = x,y,z,w
# return [roll,pitch,yaw]
def toEulerAngles(q):
    '''Convert quaternion to euler angles

    Parameters
    ----------
    q   :   np.array or list

    Returns
    -------
    np.ndarray
        Array of 3 elements [roll, pitch, yaw]
    '''
    sinr = 2.0 * (q[3] * q[0] + q[1] * q[2])
    cosr = 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1])
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (q[3] * q[1] - q[2] * q[0])

    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny = 2.0 * (q[3] * q[0] + q[0] * q[1])
    cosy = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw])


def posesFromQuaternionToRPY(poses):
    '''Batch-convert a set of poses from quaternions to euler angles.'''
    poses_xyzrpy = []
    for idx in range(0, len(poses)):
        pose = np.zeros(6)
        pose[0:3] = poses[idx, 0:3]
        pose[3:6] = toEulerAngles(poses[idx, 3:7])
        poses_xyzrpy.append(pose)

    return np.array(poses_xyzrpy)


def resize_to_multiple(images, multiples):
    '''Resize a batch of images in the height and width dimensions so their size are an integer
    multiple of some value.

    Parameters
    ----------
    images  :   tf.Tensor
                Tensor of shape [batch, height, width, channels]
    multiples   :   int or tuple
                    The value/s that should evenly divide the resized image's dimensions

    Returns
    -------
    tf.Operation
        Tensorflow op for resizing images
    '''
    from tensorflow.image import resize_images
    _, h, w, _ = images.get_shape()
    # if only one multiple, assume it's the value to use for all dims
    if not isinstance(multiples, tuple):
        multiples = (multiples, multiples)
    new_h, new_w = [int(ceil(input_shape[0] / multiples[0])),
                    int(ceil(input_shape[1] / multiples[1]))]
    return resize_images(images, [new_h, new_w])


def image_pairs(image_sequence, sequence_length):
    '''Generate sequences of stacked pairs of images where two 3-channel images are merged to on
    6-channel image. If the image sequence length is not evenly divided by the sequence length,
    fewer than the total number of images will be yielded.

    .. note:: Deprecated
        This function is deprecated by :py:class:`DataManager`


    Parameters
    ----------
    image_sequence  :   np.ndarray
                        Array of shape (num, h, w, 3)
    sequence_length  :  int
                        Number of elements (6-channel imgs) yielded each time

    Returns
    -------
    np.ndarray
        Array of shape (sequence_length, h, w, 6)
    '''
    N, h, w, c = image_sequence.shape
    for idx in range(0, N, sequence_length):
        stacked_indices = np.empty((sequence_length - 1) * 2, dtype=np.uint8)
        batch_indices = np.arange(sequence_length - 1) + idx
        stacked_indices[0::2] = batch_indices
        stacked_indices[1::2] = batch_indices + 1
        # stacked is [img0, img1, img1, img2, img2, img3, ...]
        # stacked.shape = (sequence_length * 2, h, w, c)
        stacked = image_sequence[stacked_indices, ...]

        # return array stacks every 2 images together and thus has 6 channels per image, each image
        # appears twice
        ret = np.empty((sequence_length, h, w, 2 * c), dtype=stacked.dtype)

        indices = np.arange(0, sequence_length - 1)
        ret[indices, ..., 0:3] = stacked[indices * 2]
        ret[indices, ..., 3:6] = stacked[indices * 2 + 1]

        assert (ret[0, ..., :3] == image_sequence[0]).all()
        assert (ret[0, ..., 3:] == image_sequence[1]).all()

        yield ret


def compute_rgb_mean(image_sequence):
    '''Compute the mean over each channel separately over a set of images.

    Parameters
    ----------
    image_sequence  :   np.ndarray
                        Array of shape (N, h, w, c) or (h, w, c)
    '''
    if image_sequence.ndim == 4:
        _, h, w, c = image_sequence.shape
    if image_sequence.ndim == 3:
        h, w, c = image_sequence.shape
    # compute mean separately for each channel
    # somehow this expression is buggy, so we must do it manually
    # mode = image_sequence.mean((0, 1, 2))
    mean_r = image_sequence[..., 0].mean()
    mean_g = image_sequence[..., 1].mean()
    mean_b = image_sequence[..., 2].mean()
    mean = np.array([mean_r, mean_g, mean_b])
    return mean


def convert_large_array(file_in, file_out, dtype, factor=1.0):
    '''Convert data type of an array possibly too large to fit in memory.
    This uses memory-mapped files and will therefore be very slow.

    Parameters
    ----------
    file_in :   str
                Name of the input file
    file_out    :   str
                    Name of the output file
    dtype   :   np.dtype
                Destination data type
    factor  :   float
                Scaling factor to apply to all elements
    '''
    source = np.lib.format.open_memmap(file_in, mode='r')
    dest = np.lib.format.open_memmap(file_out, mode='w+', dtype=dtype, shape=source.shape)
    np.copyto(dest, source, casting='unsafe')
    if factor != 1.0:
        np.multiply(dest, factor, out=dest)

def subtract_poses(pose_x, pose_y):
    pose_diff = np.subtract(pose_x, pose_y)
    pose_diff[..., 3:6] = np.arctan2(np.sin(pose_diff[..., 3:6]), np.cos(pose_diff[..., 3:6]))
    return pose_diff

import os
from glob import glob
from os.path import join
from skimage.transform import resize


class DataManager(object):
    def __init__(self,
                 dataset_path='data/dataset1/',
                 batch_size=10,
                 train_test_ratio=0.7,
                 sequence_length=10,
                 debug=False,
                 dtype=np.float32,
                 resize_to_width=None):

        if not os.path.exists(dataset_path):
            raise ValueError(f'Path {dataset_path} does not exist.')

        self.dtype        = dtype
        self.debug        = debug
        self.dataset_path = dataset_path
        self.images_path  = join(dataset_path, 'images')
        self.poses_path   = join(dataset_path, 'poses')

        image_files = glob(join(self.images_path, '*.npy'))
        self.N      = len(image_files)
        self.NTrain = int(self.N * train_test_ratio)
        self.NTest  = self.N - self.NTrain

        self.num_dec_file = sum(c.isdigit() for c in os.path.basename(image_files[0]))

        self.image_file_template = join(self.images_path, 'image%0') + f'{self.num_dec_file}d.npy'
        self.pose_file_template  = join(self.poses_path, 'pose%0') + f'{self.num_dec_file}d.npy'

        init_image = self.loadImage(0)
        if resize_to_width is not None:
            width_ratio = resize_to_width / init_image.shape[1]
            scaled_height = np.floor(init_image.shape[0] * width_ratio)
            init_image = resize(init_image, output_shape=(scaled_height, resize_to_width))

        self.H = init_image.shape[0]
        self.W = init_image.shape[1]
        self.C = init_image.shape[2]

        self.sequence_length = sequence_length

        self.batch_size = batch_size
        # additional frames needed depending on sequence length
        self.batch_images = np.empty(
            [self.batch_size, self.sequence_length, self.H, self.W, self.C * 2],
            dtype=dtype
        )

        self.batch_poses = np.empty([self.batch_size, self.sequence_length, 6])

        if self.debug:
            print(f'DataManager found {self.N} images and poses in dataset.')
            print(f'Image shape: {self.getImageShape()}')

    def getImageShape(self):
        return (self.H, self.W, self.C)

    def numTestBatches(self):
        return self.NTest

    def numTrainBatches(self):
        return self.NTrain

    def __len__(self):
        return self.N

    def batches(self):
        # 1D length of batch_size times sequence length
        chunk_size = self.batch_size * self.sequence_length
        for batch_start_idx in range(0, self.NTrain, chunk_size):
            record_in_batch = 0
            for sequence_start_idx in range(batch_start_idx, batch_start_idx + chunk_size,
                                            self.sequence_length):

                sequence_end_idx = sequence_start_idx + self.sequence_length + 1
                if sequence_end_idx >= self.NTrain:
                    return
                image_indices = np.arange(sequence_start_idx, sequence_end_idx)

                # generate sequences
                images = self.loadImages(image_indices)
                poses  = self.loadPoses(image_indices)

                self.batch_images[record_in_batch, ..., :3] = images[:-1]
                self.batch_images[record_in_batch, ..., 3:] = images[1:]

                # subtract first pose from all
                # absolute pose to first pose
                self.batch_poses[record_in_batch, ...] = subtract_poses(poses[1:], poses[0])
                record_in_batch += 1

            yield self.batch_images, self.batch_poses

    def test_batches(self):
        # 1D length of batch_size times sequence length
        chunk_size = self.batch_size * self.sequence_length
        for batch_start_idx in range(self.NTrain-1, self.N, chunk_size):
            record_in_batch = 0
            for sequence_start_idx in range(batch_start_idx, batch_start_idx + chunk_size,
                                            self.sequence_length):

                sequence_end_idx = sequence_start_idx + self.sequence_length + 1
                if sequence_end_idx >= self.N:
                    return
                image_indices = np.arange(sequence_start_idx, sequence_end_idx)

                # generate sequences
                images = self.loadImages(image_indices)
                poses  = self.loadPoses(image_indices)

                self.batch_images[record_in_batch, ..., :3] = images[:-1]
                self.batch_images[record_in_batch, ..., 3:] = images[1:]

                # subtract first pose from all
                # absolute pose to first pose
                self.batch_poses[record_in_batch, ...] = subtract_poses(poses[1:], poses[0])
                record_in_batch += 1

            yield self.batch_images, self.batch_poses

    def loadImage(self, id):
        img = np.squeeze(np.load(self.image_file_template % id))
        return img

    def saveImage(self, id, img):
        np.save(self.image_file_template % id, img)

    def loadImages(self, ids):
        num_images = len(ids)
        images     = np.empty([num_images, self.H, self.W, self.C], dtype=self.dtype)
        for idx in range(0, num_images):
            # right colors:
            img = self.loadImage(ids[idx])
            if img.shape != (self.H, self.W, self.C):
                images[idx] = resize(img, output_shape=(self.H, self.W), preserve_range=True)
            else:
                images[idx] = img
        return images

    def loadPose(self, id):
        return np.load(self.pose_file_template % id)

    def savePose(self, id, pose):
        np.save(self.pose_file_template % id , pose)

    def loadPoses(self, ids):
        num_poses = len(ids)
        poses     = np.empty([num_poses, 6])
        for idx in range(0, num_poses):
            poses[idx] = self.loadPose(ids[idx])
        return poses


class OptimizerSpec(dict):
    '''Encapsulate all the info needed for creating any kind of optimizer. Learning rate scheduling
    is fixed to exponential decay

    Attributes
    ----------
    step_counter    :   Variable
                        Counter to be passed to optimizer#minimize() so it gets incremented during
                        each update
    learning_rate   :   tf.train.piecewise_constant
                        Learning rate of the optimizer (for later retrieval)

    '''

    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
        kind    :   str
                    Name of the optimizer
        learning_rate   :   float
                            Base learning rate used
        name    :   str
                    Optional name for the piecewise_constant operation
        momentum    :   float
                        Optional momentum for momentum optimizers
        use_nesterov    :   bool
                            Nesterov flag for momentum optimizer
        steps   :   int (optional)
                    Exponential decay steps
        decay   :   int (optional)
                    Exponential decay rate
        '''
        if not 'kind' in kwargs:
            raise ValueError('No optimizer name given')
        if not 'learning_rate' in kwargs:
            raise ValueError('No base learning_rate given')
        self.update(kwargs)
        import tensorflow as tf
        self.step_counter  = tf.Variable(0, trainable=False, dtype=tf.int32, name='step_counter')
        rate               = kwargs['learning_rate']
        # use exponential_decay
        if 'steps' in kwargs and 'decay' in kwargs:
            steps              = kwargs.get('steps')
            decay              = kwargs.get('decay')
            self.learning_rate = tf.train.exponential_decay(rate, self.step_counter, steps, decay)
        else:   # plain learning
            self.learning_rate = rate

    def create(self):
        '''Build the Optimizer object from the properties

        Return
        ------
        tf.train.Optimizer
            Ready-made optimizer
        '''
        kind          = self['kind']
        learning_rate = self.learning_rate
        name          = self.get('name', 'optimizer')
        optimizer_cls = OptimizerSpec.get_optimizer(kind)
        if kind in ['Momentum', 'RMSProp']:
            # only those two use momentum param
            try:
                momentum = self['momentum']
            except KeyError:
                raise ValueError('Momentum parameter is necessary for MomentumOptimizer')
            if kind == 'Momentum':
                if 'use_nesterov' in self:
                    use_nesterov = self['use_nesterov']
                else:
                    use_nesterov = False
                return optimizer_cls(learning_rate, momentum, use_nesterov, name=name)
            else:
                return optimizer_cls(learning_rate, momentum, name=name)
        else:
            return optimizer_cls(learning_rate, name=name)

    def __str__(self):
        key_val_str = ', '.join(str(k) + '=' + str(v) for k, v in self.items())
        return f'<Optimizer: {key_val_str}>'

    @staticmethod
    def get_optimizer(name):
        import tensorflow as tf
        if isinstance(name, tf.train.Optimizer):
            return name
        else:
            return getattr(tf.train, name + 'Optimizer')


def conv_layer(input, channels_out, kernel_width, strides, activation, kernel_initializer,
               bias_initializer, use_bias=True, padding='SAME',
               var_names=(None, None), trainable=True):
    '''Create a convolutional layer with activation function and variable
    initialisation.

    Parameters
    ----------
    input   :   tf.Variable
                Input to the layer
    channels_out    :   int
                    Number of output feature maps
    kernel_width  :   int
                Size of the convolution filter
    strides :   tuple or int
                Strides
    activation  :   function
                    Activation function
    use_bias    :   bool
    padding :   str
                'SAME' or 'VALID'
    var_names   :   tuple
                Names of the weight and bias variables
    trainable   :   bool

    Returns
    -------
    tf.Variable
            The variable representing the layer activation

    '''
    import tensorflow as tf
    if not activation:
        activation = tf.identity
    kernel_name = var_names[0] or 'kernels'
    bias_name = var_names[1] or 'bias'
    _, h, w, channels_in = input.shape
    if isinstance(strides, int):
        strides = (1, strides, strides, 1)
    kernels = tf.get_variable(shape=(kernel_width, kernel_width, channels_in, channels_out),
                              initializer=kernel_initializer, name=kernel_name, trainable=trainable)
    if use_bias:
        bias_shape = (channels_out,)
        biases = tf.get_variable(shape=bias_shape, initializer=bias_initializer, name=bias_name,
                                 trainable=trainable)
    conv = tf.nn.conv2d(
        input,
        kernels,
        strides,
        padding=padding)
    if use_bias:
        return activation(conv + biases)
    else:
        return activation(conv)
