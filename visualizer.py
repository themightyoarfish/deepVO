import numpy as np
import matplotlib.pyplot as plt

class Visualizer(object):

    """This class visualizes the output of the network"""

    def __init__(self):
        super(Visualizer, self).__init__()
        # Define initial position to be (0,0)
        self.outputs  = []
        self.labels   = []
        self.position = 0
        self.reset_plot()

    def add_data(self, output, label):
        self.outputs.append(np.copy(output))
        self.labels.append(np.copy(label))

    def plot(self):
        # If there is no or no new information do not plot.
        if len(self.outputs) == 0 or self.position == len(self.outputs):
            return

        # Extract only the relevant data the has not yet been plotted
        output_to_plot = np.asarray(self.outputs[self.position:])
        label_to_plot  = np.asarray(self.labels[self.position:])

        # Only plot the positions
        output_to_plot = output_to_plot[:, :2]
        label_to_plot  = label_to_plot[:, :2]

        # Update the data pointer
        self.position = len(self.outputs)

        self.ax.plot(output_to_plot[:, 0], output_to_plot[:, 1], 'ro-')
        self.ax.plot(label_to_plot[:, 0], label_to_plot[:, 1], 'g^-')
        plt.pause(2)

    def reset_plot(self):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        # Clear the contents
        del self.outputs[:]
        del self.labels[:]

    def save_plot(self, path):
        pass


def main():
    vis = Visualizer()
    # generate pose data
    label = np.zeros(6)
    for i in range(20):
        # imitate movement by adding random numbers
        label += np.random.randn(6)
        output = label + np.random.randn(6)*0.5

        # slightly change the output from the groundtruth position
        vis.add_data(output, label)
    vis.plot()

    for i in range(20):
        # imitate movement by adding random numbers
        label += np.random.randn(6)
        output = label + np.random.randn(6)*0.1

        # slightly change the output from the groundtruth position
        vis.add_data(output, label)
    vis.plot()


if __name__ == "__main__":
    main()
