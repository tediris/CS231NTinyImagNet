import numpy as np
import tensorflow as tf

def maxout(x, num_units, k, name):
    # assume that x is a batch size by input_dim tensor
    shape = inputs.get_shape().as_list()
    input_dim = shape[1]
    w = tf.get_variable(name + "/maxoutw", shape=[input_dim, num_units, k], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name + "/maxoutb", shape=[num_units, k])
    w_reshape = tf.reshape(w, (input_dim, num_units * k))
    b_reshape = tf.reshape(b, (num_units * k,))
    pre_max = tf.matmul(x, w_reshape) + b_reshape
    pre_reshaped = tf.reshape(pre_max, (-1, num_units, k))
    output = tf.reduce_max(pre_reshaped, axis=2)
    return output
