import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

class PerformanceVisualizer(object):

    """This class visualizes the performance of the network by plotting the percentage of the offset"""

    def __init__(self):
        super(PerformanceVisualizer, self).__init__()
        # contains 1 to n np.arrays that store the offset percentages for a single batch
        self.trans_diffs = []
        self.rot_diffs   = []

    # in model --> x: Label, y: Prediction
    def add_rotation_batch(self, prediction_batch, label_batch):
        self.rot_diffs.append(calculate_MSE_percentage(prediction_batch, label_batch))

    def add_translation_batch(self, prediction_batch, label_batch):
        self.trans_diffs.append(calculate_MSE_percentage(prediction_batch, label_batch))

    def calculate_MSE_percentage(self, prediction_batch, label_batch):
        # Get the offset between prediction and labels
        diff = prediction_batch - label_batch
        # Calculate the norm2 of those differences
        errors = LA.norm(diff, axis=2, ord=2)
        # Sum the errors in order to normalize them later again. The array is of size N=batchsize
        # as it contains the summed errors for every sequence.
        summed_errors = np.sum(errors, axis=1)

        # get the driving distance of the label path
        # Shift the batch by +1 in order to perform a pairwise subtraction next
        shifted_label_batch = np.roll(label_batch, 1, axis=1)
        # get diff between the points but ignore the first value as it is invalid bc of shifting
        diff = (shifted_label_batch - label_batch)[:,1:,:]
        # calculate the norm2 of the differences in order to get all distances between succesive
        # points
        distances = LA.norm(diff, axis=2, ord=2)
        # Now we sum it up and get an array of size N=batchsize that contains the lenghts of all
        # sequences in it
        driving_distance = np.sum(distances, axis=1)

        # Now simply normalize the errors on the sequence distance.
        return summed_errors / driving_distance

    def plot(self):
        pass

    def save_plot(self, path):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
