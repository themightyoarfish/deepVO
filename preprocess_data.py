from argparse import ArgumentParser
from utils import image_pairs, subtract_mean_rgb, convert_large_array
import numpy as np
import sys


def main():
    parser = ArgumentParser('Preprocess pluto data vor DeepVO')
    parser.add_argument('-i', '--imgs', type=str, required=True, help='Path to image npy file')
    parser.add_argument('-p', '--poses', type=str, required=True, help='Path to pose npy file')
    parser.add_argument('-f', '--to-float', required=False, default=False,
                        action='store_true', help='Convert image array to float')
    parser.add_argument('-m', '--mean-normalize', required=False, default=False,
                        action='store_true', help='Substract rgb mean from images')
    args = parser.parse_args()

    if not args.imgs.endswith('.npy') or not args.poses.endswith('.npy'):
        print(f'Both input files must be npy files.', file=sys.stderr)
        exit(1)

    if args.to_float:
        print(f'Converting {args.imgs} fo float ...')
        convert_large_array(args.imgs, f'{args.imgs[:-4]}_float.npy', np.float32, 1/255)
        print('Done')

    if args.mean_normalize:
        print(f'Mean-normalizing {args.imgs} ...')
        images = np.load(args.imgs)     # shape (N, w, h)
        print(f'Loaded {(images.size * 4) // 1000}kb of images.')
        poses = np.load(args.poses)     # shape (N, 7)

        if not images.shape[0] == poses.shape[0]:
            print('Different number of images and poses', file=sys.stderr)
            exit(2)

        subtract_mean_rgb(images)

        np.save(f'{args.imgs[:-4]}_processed.npy', images)
        np.save(f'{args.poses[:-4]}_processed.npy', poses)
        print('Done')


if __name__ == '__main__':
    main()
