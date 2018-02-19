from . import conversions
import numpy as np

class DataManager(object):
    def __init__(self,
            path_to_images = 'data/images.npy',
            path_to_poses = 'data/poses.npy',
            batch_size = 100,
            seq_len = 2
            ):
        self.poses = np.load(path_to_poses)
        self.images = np.load(path_to_images)
        self.seq_len = seq_len
        self.batch_size = batch_size
        # additional frames needed depending on sequence length
        self.add_frames = self.seq_len - 1

        self.N = self.images.shape[0]
        self.H = self.images.shape[1]
        self.W = self.images.shape[2]
        self.C = self.images.shape[3]

        self.image_indices = np.arange(batch_size + self.add_frames)


        self.image_stack_batch = np.zeros(
                [self.batch_size, self.H, self.W, self.C*self.seq_len]
            )



    def poseContainsQuaternion(self):
        return self.poses.shape[1] == 7

    def convertPosesToRPY(self):
        self.poses = conversions.posesFromQuaternionToRPY(self.poses)

    def getNextBatch(self):
        pass

    def batches(self):

        for batch_idx in range(0, self.N, len(self.image_indices) ):
            # creating batch

            # TODO: better
            if batch_idx + len(self.image_indices) > self.N:
                break

            image_indices_global = self.image_indices + batch_idx

            # for seq_len = 3
            # image_indices_global[:-2], image_indices_global[1:-1], image_indices_global[2:]

            for i in range(0, self.seq_len):
                if(i == self.seq_len - 1):
                    self.image_stack_batch[..., self.C*i:self.C*(i+1)] = self.images[ image_indices_global[i:] ]
                else:
                    self.image_stack_batch[..., self.C*i:self.C*(i+1)] = self.images[ image_indices_global[i:-(self.add_frames-i) ] ]

            yield self.image_stack_batch





