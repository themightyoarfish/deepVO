#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import sys

from utils import DataManager, compute_rgb_mean


def main():
    parser = ArgumentParser('Preprocess pluto data vor DeepVO')
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path to dataset (a folder with "images" and "poses" subfolders.)')
    parser.add_argument('-f', '--to-float', required=False, default=False,
                        action='store_true', help='Convert images array to float')
    parser.add_argument('-m', '--mean-normalize', required=False, default=False,
                        action='store_true', help='Subtract rgb mean from images')
    args = parser.parse_args()

    data_manager = DataManager(args.data, dtype=np.float32, batch_size=1, sequence_length=1)
    if args.to_float:
        to_float(data_manager)

    if args.mean_normalize:
        mean_normalize(data_manager)


def to_float(data_manager):
    N = len(data_manager)
    print(f'Converting {data_manager.dataset_path}images/*.npy fo float ...')
    for idx in range(N):

        # print progress
        if idx % 10 == 0:
            print(f'\r{idx+1:4d}/{N}', end='')

        img = data_manager.loadImage(idx)
        data_manager.saveImage(idx, img.astype(np.float32))
    print('\nDone')


def mean_normalize(data_manager):
    N = len(data_manager)
    print(f'Mean-normalizing {data_manager.dataset_path}images/*.npy ...')
    mean_accumlator = np.zeros((3,), dtype=np.float32)

    # run over entire dataset to compute mean (fucking inefficient but I have other shit to do)
    for idx in range(N):
        img = data_manager.loadImage(idx)
        mean_accumlator += compute_rgb_mean(img)

    mean_accumlator /= N
    print(f'Mean: {mean_accumlator}')
    for idx in range(N):

        if idx % 10 == 0:
            print(f'\r{idx+1:4d}/{N}', end='')
        img = data_manager.loadImage(idx)
        data_manager.saveImage(idx, img - mean_accumlator)
    print('\nDone')





if __name__ == '__main__':
    main()
