import sys
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf

from model import VOModel
from utils import DataManager


def main():
    parser = ArgumentParser('Test')
    parser.add_argument('-i', '--imgs', type=str, required=True, help='Path to image preprocessed npy file')
    parser.add_argument('-p', '--poses', type=str, required=True, help='Path to pose preprocessed npy file')
    args = parser.parse_args()

    sequence_length = 10
    dm = DataManager(path_to_images=args.imgs,
                     path_to_poses=args.poses,
                     batch_size=1,
                     seq_len=sequence_length)

    image_shape = dm.getImageShape()
    memory_size = 1000

    # create model
    model = VOModel(image_shape, memory_size, sequence_length)

    if dm.poseContainsQuaternion():
        dm.convertPosesToRPY()

    with tf.Session() as session:
        for images, labels in dm.batchesWithSequences():
            session.run(tf.global_variables_initializer())
            model.get_rnn_output(session, images, labels)


if __name__ == '__main__':
    main()
