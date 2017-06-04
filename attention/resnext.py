import tensorflow as tf

def resnext_layer(x, is_training, num_filters, num_layers=2, name=None):

    input_layer = x
    for i in range(num_layers):
        bn = tf.layers.batch_normalization(input_layer, axis=3, training=is_training)
        relu = tf.nn.relu(bn)
        conv = tf.layers.conv2d(relu, num_filters, kernel_size=3, strides=(1, 1), padding='same')
        input_layer = conv

    # do the residual calculation
    # y = tf.nn.relu(x + input_layer)
    y = x + input_layer
    return y

def resnext_downsample(x, is_training, num_filters, num_layers=2, name=None):
    # weight 1
    bn1 = tf.layers.batch_normalization(x, axis=3, training=is_training)
    relu1 = tf.nn.relu(bn1)
    conv1 = tf.layers.conv2d(relu1, num_filters, kernel_size=3, strides=(1, 1), padding='same')

    input_layer = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    for i in range(num_layers):
        bn = tf.layers.batch_normalization(input_layer, axis=3, training=is_training)
        relu = tf.nn.relu(bn)
        conv = tf.layers.conv2d(relu, num_filters, kernel_size=3, strides=(1, 1), padding='same')
        input_layer = conv

    # downsample from the larger image
    input_channel = x.get_shape().as_list()[-1]
    next_im_size = num_filters // 4
    pooled_input = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [next_im_size, next_im_size]])

    # do the residual calculation
    # y = tf.nn.relu(padded_input + input_layer)
    y = padded_input + input_layer
    # y = input_layer
    return y

def resnext_block(x, is_training, num_filters, num_layers=2, num_modules=3, name=None, downsample=True):
    # output has double the number of filters as the input
    for module_num in range(num_modules):
        x = resnext_layer(x, is_training, num_filters, num_layers, "residual" + str(module_num))
    if downsample:
        x = resnext_downsample(x, is_training, num_filters * 2, num_layers, "residual_down")
    return x

def resnext_model(x, is_training):
    '''
    x is the raw input image, 64x64x3
    '''
    # the first scale
    num_filters = 64

    # do the initial convolution
    x = tf.layers.conv2d(x, 64, 7, strides=(1, 1), padding='same')
    # 64
    # x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    # 32

    # perform the residual blocks
    x = resnext_block(x, is_training, 64, 2, 3, "res_block_1")
    # 32
    x = resnext_block(x, is_training, 128, 2, 3, "res_block_2")
    # 16
    x = resnext_block(x, is_training, 256, 2, 3, "res_block_3")
    # 8
    x = resnext_layer(x, is_training, 512, 2, "res_last_1")
    x = resnext_layer(x, is_training, 512, 2, "res_last_2")

    x = tf.nn.avg_pool(x,[1, 8, 8, 1],strides=[1,8,8,1],padding='VALID')
    return x
