import sys
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf

from model import VOModel
from utils import DataManager

from matplotlib import pyplot as plt


def main():
    parser = ArgumentParser('Test')
    parser.add_argument('-d', '--dataset', type=str, required=False, help='Path to dataset folder')
    args = parser.parse_args()

    sequence_length = 10
    batch_size = 50

    dm = DataManager(
                dataset_path='args.dataset',
                batch_size=batch_size,
                seq_len=sequence_length,
                debug=True)

    image_shape = dm.getImageShape()
    memory_size = 1000

    # create model
    model = VOModel(image_shape, memory_size, sequence_length)

    with tf.Session() as session:
        for images, labels in dm.batchesWithSequences():
            session.run(tf.global_variables_initializer())
            model.get_rnn_output(session, images, labels)


if __name__ == '__main__':
    main()
