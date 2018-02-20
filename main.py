import sys
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf

from model import VOModel
from utils import image_pairs


def main():
    parser = ArgumentParser('Test')
    parser.add_argument('-i', '--imgs', type=str, required=True, help='Path to image preprocessed npy file')
    parser.add_argument('-p', '--poses', type=str, required=True, help='Path to pose preprocessed npy file')
    args = parser.parse_args()
    images = np.load(args.imgs)     # shape (N, w, h)
    poses = np.load(args.poses)[..., :6]     # shape (N, 7)

    _, h, w, c = images.shape
    image_shape = (h, w, c)
    memory_size = 1000
    sequence_length = 10
    input_generator = image_pairs(images, sequence_length)
    input_data, input_labels = next(input_generator), poses[1:1 + sequence_length]

    model = VOModel(image_shape, memory_size, sequence_length)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.get_cnn_output(session, input_data[np.newaxis, ...], input_labels)


if __name__ == "__main__":
    main()
