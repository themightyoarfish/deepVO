#!/usr/bin/env python3.6
'''
.. module:: main

.. moduleauthor:: Rasmus Diederichsen

'''
import sys
import numpy as np
# np.random.seed(1)
from argparse import ArgumentParser
import tensorflow as tf
# tf.set_random_seed(1)
from os.path import join
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import VOModel
from utils import OptimizerSpec
from data_manager import DataManager
from performance_visualizer import PerformanceVisualizer
from matplotlib import pyplot as plt


def make_parser():
    '''Function returning parser is necessary for sphinx-argparse'''
    tf_optimizers = {
        class_name[:-len('Optimizer')]
        for class_name in dir(tf.train)
        if 'Optimizer' in class_name and class_name != 'Optimizer'
    }
    parser = ArgumentParser('Train the DeepVO model')
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset folder')
    parser.add_argument(
        '-o',
        '--optimizer',
        required=True,
        type=str,
        choices=tf_optimizers,
        help='Optimization algorithm')
    parser.add_argument(
        '-l',
        '--learning-rate',
        required=True,
        type=float,
        help='Learning rate for the optimizer')
    parser.add_argument(
        '-b', '--batch-size', required=True, type=int, help='Batch size')
    parser.add_argument(
        '-e', '--epochs', required=True, type=int, help='Number of epochs')
    parser.add_argument(
        '-f',
        '--flownet',
        required=False,
        type=str,
        default=None,
        help='Path to pretrained flownet weights')
    parser.add_argument(
        '-m',
        '--memory-size',
        required=True,
        type=int,
        help='Size of the lstm cell memory')
    parser.add_argument(
        '-s',
        '--sequence-length',
        required=True,
        type=int,
        help='Length of the sequences used for training the RNN.')
    parser.add_argument(
        '-r',
        '--use-dropout',
        action='store_true',
        default=False,
        help='Use dropout (during training)')
    parser.add_argument(
        '-v',
        '--visualize-displacement',
        action='store_true',
        default=False,
        help='Plot the percentage of translational and rotational displacement'
    )
    parser.add_argument(
        '-w',
        '--width',
        type=int,
        required=False,
        default=0,
        help='Resize images to long edge')
    parser.add_argument(
        '-c',
        '--load-checkpoint',
        action='store_true',
        default=False,
        help='Load checkpoint from checkpoints/deepvo.ckpt')
    return parser


def main():

    args = make_parser().parse_args()

    data_manager_args = {
        'dataset_path': args.dataset,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'debug': True
    }
    if args.width != 0:
        data_manager_args['resize_to_width'] = args.width

    data_manager = DataManager(**data_manager_args)

    image_shape = data_manager.getImageShape()

    # create model
    optimizer_spec = OptimizerSpec(
        kind=args.optimizer, learning_rate=args.learning_rate)
    model = VOModel(
        image_shape,
        args.memory_size,
        args.sequence_length,
        optimizer_spec=optimizer_spec,
        use_dropout=args.use_dropout,
        use_flownet=args.flownet is not None)

    saver = tf.train.Saver()

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        if args.flownet:
            model.load_flownet(session, args.flownet)

        checkpoint_dir = 'checkpoints'
        model_filename = os.path.join(checkpoint_dir, 'deepvo.ckpt')

        if args.load_checkpoint:

            print(f'Loading {model_filename}')
            try:
                saver.restore(session, model_filename)
                print('Success.')
            except:
                print('Fail, Proceeding with random weights.')

        print('Start training...')
        best_loss = None
        losses = []
        for e in range(args.epochs):
            print(f'Epoch {e+1} of {args.epochs}')
            # reset state after each batch of consecutive sequences
            states = None
            for images, poses in data_manager.batches():
                _, _, states = model.train(
                    session, images, poses, initial_states=states)

            # test on test set. We can't push it thorugh the net at once, so be use batches and
            # compute average loss
            avg_loss = 0
            count = 0
            states = None
            for images, poses in data_manager.test_batches():
                y_t, y_r, _loss, states = model.test(
                    session, images, poses, initial_states=states)
                avg_loss += _loss
                count += 1

            avg_loss /= count
            losses.append(avg_loss)
            if not best_loss or avg_loss < best_loss:
                best_loss = avg_loss
                if not os.path.exists(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                save_path = saver.save(session, model_filename)
                print(f'Model saved in file: {model_filename}')

            print(
                f'Average test loss across {count} batches: {avg_loss:04.5f}')

            data_manager.shuffleBatches()

        f, ax = plt.subplots(1)
        ax.plot(np.arange(args.epochs), losses)
        ax.set_title('Losses with 7/3 train-test split')
        ax.set_xlabel('epoch')
        ax.set_ylabel('Test loss')
        f.savefig('losses.pdf')


if __name__ == '__main__':
    main()
