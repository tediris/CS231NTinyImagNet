import tensorflow as tf

def attention_module(x, keep_prob, is_training, input_depth, output_depth, name):
    wconv = tf.get_variable(name + "/wconv", shape=[3, 3, input_depth, output_depth], initializer=tf.contrib.layers.xavier_initializer())
    bconv = tf.get_variable(name + "/bconv", shape=[output_depth])

    mask_w = tf.get_variable(name + "/wmask", shape=[7, 7, input_depth, output_depth], initializer=tf.contrib.layers.xavier_initializer())
    mask_b = tf.get_variable(name + "/bmask", shape=[output_depth])

    convolution = tf.nn.conv2d(x, wconv, strides=[1, 1, 1, 1], padding='SAME') + bconv
    mask = tf.nn.conv2d(x, mask_w, strides=[1, 1, 1, 1], padding='SAME') + mask_b
    sigmoid = tf.sigmoid(mask)
    relu = tf.nn.relu(convolution)
    spatial_batch = tf.layers.batch_normalization(relu, axis=3, training=is_training)
    # 2x2 max pooling, stride 2
    pool = tf.nn.max_pool(spatial_batch, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    mask_pool = tf.nn.avg_pool(sigmoid, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    y_out = tf.nn.dropout(pool, keep_prob) * mask_pool
    return y_out
