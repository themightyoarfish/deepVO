# A TensorFlow implementation of _DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks_

This is our submission for the ANN with TensorFlow course, winter 2017. **Please note that this implementation does not seem entirely correct. Convergence was observed only on a dataset with random moves forwards and backwards, without rotation.**

## Data Acquisition
In order to make use of the full 720 resolution of the LifeCam 3000, you must do two things
- Tell the device driver to use this resolution via `v4l2-ctl --set-fmt-video=width=1280,height=720,pixelformat=1` (the pixel format is probably not important, but you may need to adjust the ros node accordingly)
- In the `usb_cam_node`, set height and width parameters appropriately.

## Data Preprocessing
### Bagfile conversion
The first thing to do is to convert the rosbag sensor recordings with the
conversion tool lik this
```
bag_to_cam_pose_data -b <file>.bag  -d <outdir> -x -P
```
The `-P` flag is to dump one npy file for each image and pose. The `-x`
flag is for writing float image arrays instead of uint8.
This will create `images` and `poses` folders inside the chosen directory.
### Further preprocessing
Use the `preprocess_data.py` script to prepare the data for our network
* with `-d <path-to-data>` you give it the path where the `images/` and
  `poses/` folders are located. *All modifications are done in-place*
* `-f` will map the images to (0, 1)
* `-m` will subtract the mean (over the entire set) from each image
* `-p` will add Pi to all pose angles. The robot's EKF output is in the
  range (-pi, pi), but we want (0, 2pi)

## Potential Problems
- We are not sure if the timestamps of pose and camera messages are correct and thus whether the training data is good enough
- We have no control over the exposure time of the camera. Auto-exposure differences while driving around might make the problem more difficult
