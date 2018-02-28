#!/usr/bin/env python

import sys
import numpy as np
np.random.seed(1)
from argparse import ArgumentParser
import tensorflow as tf
tf.set_random_seed(1)

from model import VOModel
from utils import DataManager, OptimizerSpec

from matplotlib import pyplot as plt


def main():
    tf_optimizers = {class_name[:-len('Optimizer')] for class_name in dir(tf.train) if 'Optimizer'
            in class_name and class_name != 'Optimizer'}
    parser = ArgumentParser('Train the DeepVO model')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('-o', '--optimizer', required=True, type=str,
            choices=tf_optimizers, help='Optimization algorithm')
    parser.add_argument('-l', '--learning-rate', required=True, type=float,
            help='Learning rate for the optimizer')
    parser.add_argument('-b', '--batch-size', required=True, type=int,
            help='Batch size')
    parser.add_argument('-e', '--epochs', required=True, type=int,
            help='Number of epochs')
    parser.add_argument('-f', '--flownet', required=False, type=str, default=None,
            help='Path to pretrained flownet weights')
    parser.add_argument('-m', '--memory-size', required=True, type=int,
            help='Size of the lstm cell memory')
    parser.add_argument('-s', '--sequence-length', required=True, type=int,
            help='Length of the sequences used for training the RNN.')
    args = parser.parse_args()

    dm = DataManager(
                dataset_path=args.dataset,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                debug=True)

    image_shape = dm.getImageShape()

    # create model
    model = VOModel(image_shape, args.memory_size, args.sequence_length, args.batch_size,
                    optimizer_spec=OptimizerSpec(kind=args.optimizer, learning_rate=args.learning_rate))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for _ in range(args.epochs):
            states = None
            for images, poses in dm.batches():
                _, loss, states = model.train(session, images, poses, initial_states=states)
                print(f'loss={loss}')


if __name__ == '__main__':
    main()
