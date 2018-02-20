import sys
import numpy as np
from argparse import ArgumentParser

from model import VOModel
import tensorflow as tf

from utils import DataManager


def main():
    parser = ArgumentParser('Test')
    parser.add_argument('-i', '--imgs', type=str, required=True, help='Path to image preprocessed npy file')
    parser.add_argument('-p', '--poses', type=str, required=True, help='Path to pose preprocessed npy file')
    args = parser.parse_args()
    images = np.load(args.imgs)     # shape (N, w, h)
    poses = np.load(args.poses)     # shape (N, 7)

    batch_size = 50
    _, h, w, c = images.shape
    image_shape = (h, w, c)
    memory_size = 1000
    sequence_length = 10

    dm = DataManager(path_to_images=args.imgs,
                 path_to_poses=args.poses,
                 batch_size=50,
                 seq_len=10)

    if dm.poseContainsQuaternion():
        dm.convertPosesToRPY()

    for images, labels in dm.batchesWithSequences():
        print(images.shape)
        print(labels.shape)

    #model = VOModel(image_shape, memory_size, sequence_length, batch)
    #with tf.Session() as session:
    #    zero_state = model.get_zero_state(session, batch_size)


if __name__ == "__main__":
    main()
