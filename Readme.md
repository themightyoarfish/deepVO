# A TensorFlow implementation of _DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks_

This is our submission for the ANN with TensorFlow course, winter 2017.

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
Use the `preprocess_data.py` script with `-d <path-to-data>` and `-m` to
subtract the mean value channel wise. Changes will be written back to the
original files.
