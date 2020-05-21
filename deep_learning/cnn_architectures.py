"""
File:   deep_learning.cnn_architectures.py

Given starting point `input_t` each provided function returns a full CNN architecture. Currently, there
are two options of whether use batch normalization and whether to use learned color transformations
which is realized as additional conv layers at the beginning [Myshkin2017]

Author:
    Jan Hering (BIA/CMP)
    jan.hering@fel.cvut.cz

"""
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import dropout, fully_connected
from tflearn.layers.normalization import batch_normalization

from deep_learning import cnn_utils


def create_vgg16_network( input_t, num_classes,
                          normalize_batch=True,
                          add_color_transfer=False):

    if add_color_transfer:
        x = cnn_utils.color_transform_layers(input_t)
    else:
        x = input_t

    x = conv_2d(x, 64, 3, activation='relu', scope='conv1_1')
    x = conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = conv_2d(x, 256, 1, activation='relu', scope='conv3_3')
    x = max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = conv_2d(x, 512, 1, activation='relu', scope='conv4_3')
    x = max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = conv_2d(x, 512, 1, activation='relu', scope='conv5_3')
    x = max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = fully_connected(x, 4096, activation='relu', scope='fc6')
    if normalize_batch:
        x = batch_normalization(x, name='bn_fc6')

    x = dropout(x, 0.5, name='dropout1')

    x = fully_connected(x, 4096, activation='relu', scope='fc7')
    if normalize_batch:
        x = batch_normalization(x, name='bn_fc7')

    x = dropout(x, 0.5, name='dropout2')

    x = fully_connected(x, num_classes, activation='softmax', scope='fc8')

    return x


def create_alexnet_network( input_t, num_classes,
                            add_color_transfer=False):

    if add_color_transfer:
        x = cnn_utils.color_transform_layers(input_t)
    else:
        x = input_t

    x = conv_2d(x, 96, 11, strides=4, activation='relu')
    x = max_pool_2d(x, 3, strides=2)

    x = conv_2d(x, 256, 5, activation='relu')
    x = max_pool_2d(x, 3, strides=2)

    x = conv_2d(x, 384, 3, activation='relu')
    x = conv_2d(x, 384, 3, activation='relu')
    x = conv_2d(x, 256, 3, activation='relu')
    x = max_pool_2d(x, 3, strides=2)

    x = fully_connected(x, 2048, activation='tanh')
    x = batch_normalization(x, name='batch_fc')
    x = dropout(x, 0.5)

    x = fully_connected(x, 2048, activation='tanh')
    x = batch_normalization(x, name='batch_fc2')
    x = dropout(x, 0.5)

    x = fully_connected(x, num_classes, activation='softmax')

    return x


def create_resnext_network( input_t, num_classes,
                            resnext_n, add_color_transfer):

    if add_color_transfer:
        x = cnn_utils.color_transform_layers(input_t)
    else:
        x = input_t

    x = conv_2d(x, 16, 3, regularizer='L2', weight_decay=0.0001)
    x = tflearn.resnext_block(x, resnext_n, 16, 32)
    x = tflearn.resnext_block(x, 1, 32, 32, downsample=True)
    x = tflearn.resnext_block(x, resnext_n - 1, 32, 32)
    x = tflearn.resnext_block(x, 1, 64, 32, downsample=True)
    x = tflearn.resnext_block(x, resnext_n - 1, 64, 32)
    x = tflearn.batch_normalization(x)
    x = tflearn.activation(x, 'relu')
    x = tflearn.global_avg_pool(x)

    x = fully_connected(x, num_classes, activation='softmax')

    return x


def create_simple_network( input_t, num_classes ):

    network = conv_2d(input_t, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')

    return network