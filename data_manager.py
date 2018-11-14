import os
from glob import glob
from os.path import join
import numpy as np
from skimage.transform import resize


class DataManager(object):
    '''DataManager class for training and test data handling.

    Attributes
    ----------
    dataset_path    :   str
                        Path to directory containing the test images and poses
    target_poses    :   tf.Placeholder
                        Float placeholder of shape (batch_size, sequence_length, 6) with 3
                        translational and 3 rotational components
    batch_size      :   int
                        Batch size of requested data
    train_test_ratio:   float
                        Train data size to test data size ratio
    sequence_length :   int
                        Sequenth length of requested data
    debug           :   bool
                        Debug mode for additional information prints
    dtype           :   np.dtype
                        Numpy datatype of stored data
    N               :   int
                        Number of batches
    NTrain          :   int
                        Number of training batches
    NTest           :   int
                        Number of test batches
    '''

    def __init__(self,
                 dataset_path='data/dataset1/',
                 batch_size=10,
                 train_test_ratio=0.7,
                 sequence_length=10,
                 debug=False,
                 dtype=np.float32,
                 resize_to_width=None):
        '''
        Parameters
        ----------
        dataset_path    :   str
                            Path to directory containing the test images and poses
        target_poses    :   tf.Placeholder
                            Float placeholder of shape (batch_size, sequence_length, 6) with 3
                            translational and 3 rotational components
        batch_size      :   int
                            Batch size of requested data
        train_test_ratio:   float
                            Train data size to test data size ratio
        sequence_length :   int
                            Sequenth length of requested data
        debug           :   bool
                            Debug mode for additional information prints
        dtype           :   dtype
                            Numpy datatype of stored data
        resize_to_width :   int
                            Resize the file data
        '''

        if not os.path.exists(dataset_path):
            raise ValueError('Path ' + dataset_path + ' does not exist.')

        self.dtype = dtype
        self.debug = debug
        self.dataset_path = dataset_path
        self.images_path = join(dataset_path, 'images')
        self.poses_path = join(dataset_path, 'poses')

        image_files = glob(join(self.images_path, '*.npy'))
        self.N = len(image_files)
        self.NTrain = int(self.N * train_test_ratio)
        self.NTest = self.N - self.NTrain

        self.num_dec_file = sum(
            c.isdigit() for c in os.path.basename(image_files[0]))

        self.image_file_template = join(self.images_path, 'image%0') + str(
            self.num_dec_file) + 'd.npy'
        self.pose_file_template = join(self.poses_path, 'pose%0') + str(
            self.num_dec_file) + 'd.npy'

        init_image = self.loadImage(0)
        if resize_to_width is not None:
            width_ratio = resize_to_width / init_image.shape[1]
            scaled_height = np.floor(init_image.shape[0] * width_ratio)
            init_image = resize(
                init_image, output_shape=(scaled_height, resize_to_width))

        self.H = init_image.shape[0]
        self.W = init_image.shape[1]
        self.C = init_image.shape[2]

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.chunk_size = self.batch_size * self.sequence_length

        self.batch_positions = np.arange(0, self.N, self.chunk_size)

        self.num_batches = self.batch_positions.shape[0]
        self.num_batches_train = int(
            self.batch_positions.shape[0] * train_test_ratio)
        self.num_batches_test = self.num_batches - self.num_batches_train

        self.batch_positions_train = self.batch_positions[:self.
                                                          num_batches_train]
        self.batch_positions_test = self.batch_positions[
            self.num_batches_train:]

        if self.debug:
            print("Number of batches for training: " +
                  str(self.num_batches_train))
            print("Number of batches for testing: " +
                  str(self.num_batches_test))

        self.shuffleBatches()

        # additional frames needed depending on sequence length
        self.batch_images = np.empty(
            [
                self.batch_size, self.sequence_length, self.H, self.W,
                self.C * 2
            ],
            dtype=dtype)

        self.batch_poses = np.empty([self.batch_size, self.sequence_length, 6])

        if self.debug:
            print('DataManager found %d images and poses in dataset.' % self.N)
            print('Image shape: ' + str(self.getImageShape()))

    def getImageShape(self):
        '''Image shape of one image

        Returns
        -------
        tuple(int)
            Shape of image data
        '''
        return (self.H, self.W, self.C)

    def numTestBatches(self):
        '''Number of test batches

        Returns
        -------
        int
            Number of test batches
        '''
        return self.num_batches_test

    def numTrainBatches(self):
        '''Number of training batches

        Returns
        -------
        int
            Number of training batches
        '''
        return self.num_batches_train

    def __len__(self):
        '''Number of total images/poses

        Returns
        -------
        int
            Number of total images/poses
        '''
        return self.N

    def batches(self):
        '''Get non-overlapping training batches

        Yields
        -------
        np.ndarray
            Images with shape depending on training batch size and sequence size
        np.ndarray
            Labels with shape depending on training batch size and sequence size
        '''
        # 1D length of batch_size times sequence length
        for batch_start_idx in self.batch_positions_train:
            record_in_batch = 0
            continue_var = False
            for sequence_start_idx in range(batch_start_idx,
                                            batch_start_idx + self.chunk_size,
                                            self.sequence_length):

                sequence_end_idx = sequence_start_idx + self.sequence_length + 1
                if sequence_end_idx >= self.NTrain:
                    continue_var = True
                    break
                image_indices = np.arange(sequence_start_idx, sequence_end_idx)

                # generate sequences
                images = self.loadImages(image_indices)
                poses = self.loadPoses(image_indices)

                self.batch_images[record_in_batch, ..., :3] = images[:-1]
                self.batch_images[record_in_batch, ..., 3:] = images[1:]

                # subtract first pose from all
                # absolute pose to first pose
                self.batch_poses[record_in_batch, ...] = self._subtract_poses(
                    poses[1:], poses[0])
                record_in_batch += 1

            if not continue_var:
                yield self.batch_images, self.batch_poses

    def test_batches(self):
        '''Test batches

        Yields
        -------
        np.ndarray
            Images with shape depending on test batch size and sequence size
        np.ndarray
            Labels with shape depending on test batch size and sequence size
        '''
        # 1D length of batch_size times sequence length
        for batch_start_idx in self.batch_positions_test:
            record_in_batch = 0
            continue_var = False
            for sequence_start_idx in range(batch_start_idx,
                                            batch_start_idx + self.chunk_size,
                                            self.sequence_length):

                sequence_end_idx = sequence_start_idx + self.sequence_length + 1
                if sequence_end_idx >= self.N:
                    continue_var = True
                    break

                image_indices = np.arange(sequence_start_idx, sequence_end_idx)

                # generate sequences
                images = self.loadImages(image_indices)
                poses = self.loadPoses(image_indices)

                self.batch_images[record_in_batch, ..., :3] = images[:-1]
                self.batch_images[record_in_batch, ..., 3:] = images[1:]

                # subtract first pose from all
                # absolute pose to first pose
                self.batch_poses[record_in_batch, ...] = self._subtract_poses(
                    poses[1:], poses[0])
                record_in_batch += 1

            if not continue_var:
                yield self.batch_images, self.batch_poses

    def loadImage(self, id):
        '''Loads image with id < N

        Parameters
        -------
        id    :   int
                  id of image < N
        Returns
        -------
        np.ndarray
            Image
        '''
        img = np.squeeze(np.load(self.image_file_template % id))
        return img

    def saveImage(self, id, img):
        '''Saves image with id

        Parameters
        -------
        id    :   int
                  id of image
        img   :   np.ndarray
                  image to save
        '''
        np.save(self.image_file_template % id, img)

    def loadImages(self, ids):
        '''loads muliple images

        Parameters
        ----------
        ids :   list(int)
                List of ids to fetch
        '''
        num_images = len(ids)
        images = np.empty(
            [num_images, self.H, self.W, self.C], dtype=self.dtype)
        for idx in range(0, num_images):
            # right colors:
            img = self.loadImage(ids[idx])
            if img.shape != (self.H, self.W, self.C):
                images[idx] = resize(
                    img, output_shape=(self.H, self.W), preserve_range=True)
            else:
                images[idx] = img
        return images

    def loadPose(self, id):
        '''Ooads pose for id'''
        return np.load(self.pose_file_template % id)

    def savePose(self, id, pose):
        '''Saves pose'''
        np.save(self.pose_file_template % id, pose)

    def loadPoses(self, ids):
        '''Loads multiple poses'''
        num_poses = len(ids)
        poses = np.empty([num_poses, 6])
        for idx in range(0, num_poses):
            poses[idx] = self.loadPose(ids[idx])
        return poses

    def shuffleBatches(self):
        ''' call this after each epoch '''
        np.random.shuffle(self.batch_positions_train)
        np.random.shuffle(self.batch_positions_test)

    def _subtract_poses(self, pose_x, pose_y):
        '''Correct subtraction of two poses

        Parameters
        ----------
        pose_x  :   np.array
                    input array of poses or one pose
        pose_y  :   np.array
                    input array of poses or one pose
        return  :   np.array
                    output array of pose_x - pose_y
        '''
        pose_diff = np.subtract(pose_x, pose_y)
        pose_diff[..., 3:6] = np.arctan2(
            np.sin(pose_diff[..., 3:6]), np.cos(pose_diff[..., 3:6]))
        return pose_diff


import sys
from argparse import ArgumentParser


def make_parser():
    '''Function returning parser is necessary for sphinx-argparse'''
    parser = ArgumentParser('Test class for data manager')
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset folder')
    parser.add_argument(
        '-v',
        '--video',
        action='store_true',
        default=False,
        help='show image sequence with label information of training data')

    return parser


def main():
    import numpy as np
    from matplotlib import pyplot as plt
    args = make_parser().parse_args()

    batch_size = 10
    sequence_length = 10

    data_manager_args = {
        'dataset_path': args.dataset,
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'debug': True
    }

    data_manager = DataManager(**data_manager_args)

    image_shape = data_manager.getImageShape()

    print("Number of images in training batches: " +
          str(data_manager.numTrainBatches()))
    for images, labels in data_manager.batches():

        print(images.shape)
        print(labels.shape)

        if args.video:

            f, axarr = plt.subplots(2, 2)
            axarr[0, 0].imshow(images[0, 0, ..., :3])
            axarr[0, 0].set_title("First stack in sequence")
            axarr[0, 1].imshow(images[0, 0, ..., 3:])
            axarr[0, 1].set_title(str(labels[0, 0, ...]))
            axarr[1, 0].imshow(images[0, -1, ..., :3])
            axarr[1, 0].set_title("Last stack in sequence")
            axarr[1, 1].imshow(images[0, -1, ..., 3:])
            axarr[1, 1].set_title(str(labels[0, -1, ...]))

            plt.show()

    print("Number of images in test batches: " +
          str(data_manager.numTestBatches()))
    for images, labels in data_manager.test_batches():

        print(images.shape)
        print(labels.shape)


if __name__ == '__main__':
    main()
