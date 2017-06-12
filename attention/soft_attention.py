import tensorflow as tf
from attention.resnext import resnext_layer, resnext_block

def soft_attention_mask(input_layer, num_filters, is_training):
    input_size = input_layer.get_shape().as_list()[-2]

    pool1 = tf.nn.max_pool(input_layer, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    res1 = resnext_layer(pool1, is_training, num_filters)
    pool2 = tf.nn.max_pool(res1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    res2 = resnext_layer(pool2, is_training, num_filters)
    # res3 = resnext_layer(res2, is_training, num_filters)
    upscale1 = tf.image.resize_images(res2, [input_size // 2, input_size // 2])
    res4 = resnext_layer(upscale1, is_training, num_filters)
    upscale2 = tf.image.resize_images(res4, [input_size, input_size])

    conv1 = tf.layers.conv2d(upscale2, num_filters, kernel_size=1, strides=(1, 1), padding='same')

    mask = tf.nn.sigmoid(conv1)
    return mask

def mini_attention_mask(input_layer, num_filters, is_training):
    input_size = input_layer.get_shape().as_list()[-2]

    pool1 = tf.nn.max_pool(input_layer, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    res1 = resnext_layer(pool1, is_training, num_filters)
    res2 = resnext_layer(res1, is_training, num_filters)
    upscale2 = tf.image.resize_images(res2, [input_size, input_size])

    conv1 = tf.layers.conv2d(upscale2, num_filters, kernel_size=1, strides=(1, 1), padding='same')

    mask = tf.nn.sigmoid(conv1)
    return mask

def mini_attention_module(input_layer, num_filters, is_training):
    res1 = resnext_layer(input_layer, is_training, num_filters)

    # compute the attention mask
    attention = mini_attention_mask(res1, num_filters, is_training)

    # compute the trunk branch
    res2 = resnext_layer(res1, is_training, num_filters)
    # res3 = resnext_layer(res2, is_training, num_filters)

    combined = res2 * (1 + attention)

    res4 = resnext_layer(combined, is_training, num_filters)
    return res4

def attention_module(input_layer, num_filters, is_training):
    res1 = resnext_layer(input_layer, is_training, num_filters)

    # compute the attention mask
    attention = soft_attention_mask(res1, num_filters, is_training)

    # compute the trunk branch
    res2 = resnext_layer(res1, is_training, num_filters)
    # res3 = resnext_layer(res2, is_training, num_filters)

    combined = res2 * (1 + attention)

    res4 = resnext_layer(combined, is_training, num_filters)
    return res4

def attention_model(input_layer, is_training):
    pool1 = tf.nn.avg_pool(input_layer, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    # 32

    a1 = attention_model(pool1, 64, is_training)
    pool2 = tf.nn.max_pool(a1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    # 16
    a2 = attention_model(pool1, 128, is_training)
    pool2 = tf.nn.max_pool(a2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    # 8
    a3 = attention_model(pool1, 256, is_training)
    features = resnext_block(a3, is_training, 256)

    # pool into one thing
    output = tf.nn.avg_pool(features,[1, 8, 8, 1],strides=[1,8,8,1],padding='VALID')
    return output
