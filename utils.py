import numpy as np

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


def subtract_mean_rgb(image_sequence):
    '''Subtract the rgb mean in-place. The mean is computed and subtracted on each channel. The mean
    is computed and subtracted on each channel.

    Parameters
    ----------
    image_sequence  :   np.ndarray
                        Array of shape (N, h, w, c)
    '''
    N, h, w, c = image_sequence.shape
    # compute mean separately for each channel
    # somehow this expression is buggy, so we must do it manually
    # mode = image_sequence.mean((0, 1, 2)).astype(image_sequence.dtype)
    mean_r = image_sequence[..., 0].mean()
    mean_g = image_sequence[..., 1].mean()
    mean_b = image_sequence[..., 2].mean()

    # in order to make sure the subtraction is properly applied to each channel, we help the
    # broadcasting process by making it an array of shape (1, 1, 1, 3)
    mode = np.array([mean_r, mean_g, mean_b])
    mode = mode[np.newaxis, np.newaxis, np.newaxis, ...]

    np.subtract(image_sequence, mode, out=image_sequence)


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


import numpy as np

class DataManager(object):
    def __init__(self,
                 path_to_images='data/images.npy',
                 path_to_poses='data/poses.npy',
                 batch_size=100,
                 seq_len=2
                 ):

        self.sequence_length = seq_len
        self.poses      = np.load(path_to_poses)
        self.images     = np.load(path_to_images)
        self.seq_len    = 2
        self.batch_size = batch_size
        # additional frames needed depending on sequence length
        self.add_frames = self.seq_len - 1

        self.N = self.images.shape[0]
        self.H = self.images.shape[1]
        self.W = self.images.shape[2]
        self.C = self.images.shape[3]

        self.image_indices = np.arange(batch_size + self.add_frames)

        self.image_stack_batch = np.zeros(
            [self.batch_size, self.H, self.W, self.C * self.seq_len]
        )

        self.image_stack_batch_with_sequences = np.zeros(
            [self.batch_size, self.sequence_length, self.H, self.W, self.C * self.seq_len]
        )

    def getImageShape(self):
        return (self.H, self.W, self.C)

    def poseContainsQuaternion(self):
        return self.poses.shape[1] == 7

    def convertPosesToRPY(self):
        self.poses = posesFromQuaternionToRPY(self.poses)

    def batches(self):

        sequence_indices = np.arange(self.sequence_length)
        print(sequence_indices)

        for batch_idx in range(0, self.N, len(self.image_indices) ):
            # creating batch

            # TODO: better
            if batch_idx + len(self.image_indices) > self.N:
                break

            image_indices_global = self.image_indices + batch_idx

            # for seq_len = 3
            # image_indices_global[:-2], image_indices_global[1:-1], image_indices_global[2:]

            # build differences of poses
            # later pictures poses - first pictures poses
            diff_poses = self.poses[image_indices_global[self.add_frames:]] - self.poses[image_indices_global[:-self.add_frames] ]

            # build image sequences
            for idx in range(0, self.seq_len):
                begin = self.C * idx
                end = self.C * (idx + 1)
                if idx == self.seq_len - 1:
                    self.image_stack_batch[..., begin:end] = self.images[image_indices_global[idx:]]
                else:
                    self.image_stack_batch[..., begin:end] = self.images[image_indices_global[idx:-(self.add_frames - idx)]]

            yield self.image_stack_batch, diff_poses

    def batchesWithSequences(self, diff_poses = False):

        batch_count = 0


        chunk_size = self.batch_size * self.sequence_length



        batch_images = np.zeros(
                [self.batch_size, self.sequence_length, self.H, self.W, self.C * 2]
            )

        batch_poses = np.zeros(
                [self.batch_size, self.sequence_length, 6]
        )

        chunk_count = 0
        for chunk_point in range(self.sequence_length-1, self.N, chunk_size):

            seq_count = 0
            for seq_point in range(chunk_point, chunk_point+chunk_size, self.sequence_length):
                # print("chunk: " + str(chunk_point) + ", image_id: " + str(seq_point) )

                if seq_point >= self.N:
                    return

                image_indices = np.arange(seq_point-self.sequence_length, seq_point)

                # generate sequences
                batch_images[seq_count,...,0:3] = self.images[ image_indices - 1 ]
                batch_images[seq_count,...,3:6] = self.images[ image_indices ]

                # generate diff poses
                if diff_poses:
                    seq_poses = self.poses[ image_indices ] - self.poses[ image_indices - 1 ]
                else:
                    seq_poses = self.poses[ image_indices ]

                batch_poses[seq_count,...] = seq_poses

                seq_count = seq_count + 1

                # generate sequence and add it to batch

            chunk_count = chunk_count + 1

            yield batch_images, batch_poses

