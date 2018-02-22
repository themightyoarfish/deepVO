import numpy as np
from matplotlib import pyplot as plt


def array_from_lstm_tuple(tup):
    '''Create an array from a tuple of :py:class:``LSTMStateTuple``s.

    Parameters
    ----------
    tup :   tuple(LSTMStateTuple)
            Tuple of N_lstm ``LSTMStateTuple``s where each of the tuples has members of shape
            ``(batch_size, memory_size)``

    Returns
    -------
    np.ndarray
        Array of shape (N_lstm, 2, batch_size, memory_size) with cell and hidden states per lstm cell
        stacked together
    '''
    # one state tuple has two members of shape (batch_size, memory_size)
    N_lstm      = len(tup)
    batch_size  = tup[0].c.shape[0]
    memory_size = tup[0].c.shape[1]
    # return value
    array       = np.empty((N_lstm, 2, batch_size, memory_size))

    for lstm_idx in range(N_lstm):
        lstm_state = tup[lstm_idx]
        if not ((batch_size, memory_size) == lstm_state.c.shape == lstm_state.h.shape):
            raise ValueError('All states must have the same dimenstion.')
        array[lstm_idx, 0, ...] = lstm_state.h  # cell state
        array[lstm_idx, 1, ...] = lstm_state.c  # hidden state

    return array


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
    cosr = 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1] )
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (q[3] * q[1]  - q[2] * q[0] )

    if(np.abs(sinp) >= 1):
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny = 2.0 * (q[3] * q[0] + q[0] * q[1] )
    cosy = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2] )
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw])

def posesFromQuaternionToRPY(poses):
    '''Batch-convert a set of poses from quaternions to euler angles.'''
    poses_xyzrpy = []
    for i in range(0,len(poses)):
        pose = np.zeros(6)
        pose[0:3] = poses[i,0:3]
        pose[3:6] = toEulerAngles(poses[i,3:7])
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
    tf.op
        Tensorflow op for resizing images
    '''
    from tensorflow.image import resize_images
    _, h, w, _ = images.get_shape()
    # if only one multiple, assume it's the value to use for all dims
    if not isinstance(multiples, tuple):
        multiples = multiples * 2
    new_h, new_w = [int(ceil(input_shape[0] / multiples[0])),
                    int(ceil(input_shape[1] / multiples[1]))]
    return resize_images(images, [new_h, new_w])


def image_pairs(image_sequence, sequence_length):
    '''Generate sequences of stacked pairs of images where two 3-channel images are merged to on
    6-channel image. If the image sequence length is not evenly divided by the sequence length,
    fewer than the total number of images will be yielded.


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
    if image_sequence.ndim == 4:
        N, h, w, c = image_sequence.shape
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
    pose_diff[..., 3:6] = np.arctan2( np.sin(pose_diff[..., 3:6]), np.cos(pose_diff[..., 3:6]) )
    return pose_diff


import os
from glob import glob
from os.path import join

class DataManager(object):
    def __init__(self,
                 dataset_path='data/dataset1/',
                 batch_size=10,
                 sequence_length=10,
                 debug=False,
                 dtype=np.float32):

        if not os.path.exists(dataset_path):
            raise ValueError(f'Path {dataset_path} does not exist.')

        self.dtype        = dtype
        self.debug        = debug
        self.dataset_path = dataset_path
        self.images_path  = join(dataset_path, 'images')
        self.poses_path   = join(dataset_path, 'poses')

        image_files = glob(join(self.images_path, '*.npy'))
        self.N      = len(image_files)

        self.num_dec_file = sum(c.isdigit() for c in os.path.basename(image_files[0]))

        self.image_file_template = join(self.images_path, 'image%0') + f'{self.num_dec_file}d.npy'
        self.pose_file_template  = join(self.poses_path, 'pose%0') + f'{self.num_dec_file}d.npy'

        init_image = self.loadImage(0)

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

    def __len__(self):
        return self.N

    def batches(self):
        # 1D length of batch_size times sequence length
        chunk_size = self.batch_size * self.sequence_length
        for batch_start_idx in range(0, self.N, chunk_size):
            record_in_batch = 0
            for sequence_start_idx in range(batch_start_idx, batch_start_idx + chunk_size, self.sequence_length):

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
        return np.squeeze(np.load(self.image_file_template % id))

    def saveImage(self, id, img):
        np.save(self.image_file_template % id, img)

    def loadImages(self, ids):
        num_images = len(ids)
        images     = np.empty([num_images, self.H, self.W, self.C], dtype=self.dtype)
        for i in range(0, num_images):
            # right colors:
            images[i] = self.loadImage(ids[i])

        return images

    def loadPose(self, id):
        return np.load(self.pose_file_template % id)

    def savePose(self, id, pose):
        np.save(self.pose_file_template % id , pose)

    def loadPoses(self, ids):
        num_poses = len(ids)
        poses     = np.empty([num_poses, 6])
        for i in range(0, num_poses):
            poses[i] = self.loadPose(ids[i])
        return poses
