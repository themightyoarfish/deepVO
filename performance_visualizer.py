import numpy as np
# import matplotlib.pyplot as plt
# bc of this: https://github.com/matplotlib/matplotlib/issues/3466/#issuecomment-195899517
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA

class PerformanceVisualizer(object):

    """This class visualizes the performance of the network by plotting the percentage of the offset"""

    def __init__(self):
        super(PerformanceVisualizer, self).__init__()
        # contains 1 to n np.arrays that store the offset percentages for a single batch
        self.trans_diffs = []
        self.rot_diffs   = []

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

    # in model --> x: Label, y: Prediction
    def add_rotation_batch(self, prediction_batch, label_batch):
        self.rot_diffs.append(self.calculate_MSE_percentage(prediction_batch, label_batch))

    def add_translation_batch(self, prediction_batch, label_batch):
        self.trans_diffs.append(self.calculate_MSE_percentage(prediction_batch, label_batch))

    def plot(self, show=True):
        figure = plt.figure()
        ax = figure.add_subplot(211)
        offset = 0
        for batch in self.trans_diffs:
            batch_size = batch.shape[0]
            ax.plot(np.arange(offset, offset + batch_size), batch, '-')
            plt.title('Translational Error')
            plt.ylabel('translational error [displacement/seq_length]')
            plt.xlabel('number of sequence')
            offset += batch_size

        offset = 0
        ax = figure.add_subplot(212)
        for batch in self.rot_diffs:
            batch_size = batch.shape[0]
            ax.plot(np.arange(offset, offset + batch_size), batch, '-')
            plt.title('Rotational Error')
            plt.ylabel('rotational error [???]')
            plt.xlabel('number of sequence')
            offset += batch_size
        plt.show()

    def save_plot(self, path='.'):
        filename = path + '/performance.pdf'
        plt.savefig(filename)


def main():
    # Generate two batches of results and test the visualizer
    # Batchsize=50, seqlength=10, posesize=6
    batch_label  = np.zeros((50, 10, 6))
    batch_output = np.zeros((50, 10, 6))
    for b in range(50):
        seq_label  = np.zeros((10, 6))
        seq_output = np.zeros((10, 6))
        dev = 1 - b / 50
        for i in range(1,10):
            # imitate movement by adding random numbers
            seq_label[i, :]  = seq_label[i-1, :] + np.random.randn(6)
            seq_output[i, :] = seq_label[i, :] + np.random.randn(6) * dev
        batch_label[b, :, :]  = seq_label
        batch_output[b, :, :] = seq_output

    p = PerformanceVisualizer()
    p.add_translation_batch(batch_output[:,:,:3], batch_label[:,:,:3])
    # TODO: probably not correct data as it is rotational difference *think*
    p.add_rotation_batch(batch_output[:,:,3:], batch_label[:,:,3:])
    p.plot()


if __name__ == "__main__":
    main()
