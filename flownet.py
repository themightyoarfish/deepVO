'''
.. module:: flownet

Contains utilities for incorporating FlowNetSimple weights. The weights can be downloaded from
`here <https://s3.us-east-2.amazonaws.com/flownetdata/weights.tar.gz>`_ . This file contains all
weights for all FlowNet versions and is thus unfortunately huge.

.. data:: flownet_prefix
    The variable scope in which the flownet weights live

.. data:: flownet_kernel_suffix
    The variable name of the kernel weights

.. data:: flownet_bias_suffix
    The variable name of the bias weights

.. data:: flownet_layer_names
    Names of the 10 conv layers
'''
# variable scope as used in the checkpoints file
flownet_prefix = 'FlowNetS'

# name of the kernel weight variable
flownet_kernel_suffix = 'weights'

# name of the bias variable
flownet_bias_suffix = 'biases'

# names of the 10 conv layers
flownet_layer_names = [
    'conv1',
    'conv2',
    'conv3',
    'conv3_1',
    'conv4',
    'conv4_1',
    'conv5',
    'conv5_1',
    'conv6',
    'conv6_1',
]
