import tensorflow as tf
from tflearn.layers import conv_2d


def max_out(incoming, n_units, name='MaxOut'):
    shape = incoming.get_shape().as_list()
    axis = 3

    if shape[0] is None:
        shape[0] = -1
    #
    n_channels = shape[axis]
    # shape[axis] = n_units
    # shape += [num_channels // n_units]
    # outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    #
    # return outputs

    # input_shape = utils.get_incoming_shape(incoming)
    assert len(shape) == 4, "Incoming Tensor shape must be 4-D"

    shape[axis] = n_units
    shape += [n_channels // n_units]

    with tf.name_scope(name):
        inference = tf.reduce_max(tf.reshape(incoming, shape), -1, keep_dims=False)

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def color_transform_layers(inputs):
    # convolution 1x1
    net = conv_2d(inputs, 20, 1)
    net = max_out(net, 10)

    net = conv_2d(net, 6, 1)
    net = max_out(net, 3)

    return net
