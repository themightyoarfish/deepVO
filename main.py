import sys
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf

from model import VOModel
from utils import DataManager
from utils import DataManager2

from matplotlib import pyplot as plt


def main():
    parser = ArgumentParser('Test')
    parser.add_argument('-i', '--imgs', type=str, required=False, help='Path to image preprocessed npy file')
    parser.add_argument('-p', '--poses', type=str, required=False, help='Path to pose preprocessed npy file')
    parser.add_argument('-d', '--dataset', type=str, required=False, help='Path to dataset folder')
    args = parser.parse_args()

    sequence_length = 10
    batch_size = 50

    if args.imgs:
        dm = DataManager(path_to_images=args.imgs,
                        path_to_poses=args.poses,
                        batch_size=batch_size,
                        seq_len=sequence_length)
        if dm.poseContainsQuaternion():
            dm.convertPosesToRPY()
    elif args.dataset:
        dm = DataManager2(
                    dataset_path = 'data/dataset1/',
                    batch_size=batch_size,
                    seq_len=sequence_length,
                    debug = True
                )
    else:
        parser.print_help()
        exit()


    image_shape = dm.getImageShape()
    memory_size = 1000

    show_images = True




    for i, (images, poses) in enumerate( dm.batches(diff_poses = True) ):
        print("Batch " + str(i) )
        print(images.shape)
        print(poses.shape)

        # # very first image
        # plt.imshow(images[0,0,...,0:3])
        # plt.show()

        # # last image
        # plt.imshow(images[-1,-1,...,3:6] )
        # plt.show()



    # create model
    #model = VOModel(image_shape, memory_size, sequence_length)


    # with tf.Session() as session:
    #     for images, labels in dm.batchesWithSequences():
    #         session.run(tf.global_variables_initializer())
    #         model.get_rnn_output(session, images, labels)


if __name__ == '__main__':
    main()
