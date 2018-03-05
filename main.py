#!/usr/bin/env python3.6

import sys
import numpy as np
# np.random.seed(1)
from argparse import ArgumentParser
import tensorflow as tf
# tf.set_random_seed(1)
from os.path import join
import os.path
import os

from model import VOModel
from utils import DataManager, OptimizerSpec

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
    parser.add_argument('-r', '--use-dropout', action='store_true', default=False,
            help='Use dropout (during training)')
    parser.add_argument('-w', '--width', type=int, required=False, default=0,
            help='Resize images to long edge')
    args = parser.parse_args()
    if args.use_dropout:
        print('Use dropout')

    if args.width == 0:
        dm = DataManager(
                    dataset_path=args.dataset,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    debug=True)
    else:
        dm = DataManager(
                    dataset_path=args.dataset,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    debug=True,
                    resize_to_width=args.width)

    image_shape = dm.getImageShape()

    # create model
    optimizer_spec = OptimizerSpec(kind=args.optimizer, learning_rate=args.learning_rate)
    model = VOModel(image_shape,
                    args.memory_size,
                    args.sequence_length,
                    optimizer_spec=optimizer_spec,
                    is_training=args.use_dropout,
                    use_flownet=args.flownet is not None)

    saver = tf.train.Saver()

    model_filename = path = os.path.join(os.getcwd(), 'deepVOmodel.ckpt')

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        if args.flownet:
            model.load_flownet(session, args.flownet)

        print("search for model : " + model_filename)
        try:
            saver.restore(session, model_filename)
            print("Model restored.")
        except:
            print("No model found")

        for e in range(args.epochs):
            print(f'Epoch {e}')
            states = None
            for images, poses in dm.batches():
                _, loss, states = model.train(session, images, poses, initial_states=states)
                print(f'\tloss={loss:04.5f}')
            # Save the variables to disk.

            save_path = saver.save(session, model_filename)
            print("Model saved in file: %s" % model_filename)


if __name__ == '__main__':
    main()
