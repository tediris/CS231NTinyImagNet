import tensorflow as tf
from attention.resnext import resnext_block
from attention.soft_attention import attention_module, mini_attention_module

def residual_inception(inputs, is_training):
    # 16 x 16 x 320
    inputs = tf.layers.batch_normalization(inputs, axis=3, training=is_training)
    inputs = tf.nn.relu(inputs)

    left = tf.layers.conv2d(inputs, 128, kernel_size=1, strides=(1, 1), padding='same')

    middle = tf.layers.conv2d(inputs, 96, kernel_size=1, strides=(1, 1), padding='same')
    middle = tf.layers.conv2d(middle, 96, kernel_size=3, strides=(1, 1), padding='same')

    right = tf.layers.conv2d(inputs, 96, kernel_size=1, strides=(1, 1), padding='same')
    right = tf.layers.conv2d(right, 96, kernel_size=3, strides=(1, 1), padding='same')
    right = tf.layers.conv2d(right, 96, kernel_size=3, strides=(1, 1), padding='same')

    combined = tf.concat([left, middle, right], axis=-1)
    residual = tf.layers.conv2d(combined, 320, kernel_size=1, strides=(1, 1), padding='same')

    activated = tf.nn.relu(inputs + residual)
    return activated
    # return tf.layers.batch_normalization(activated, axis=3, training=is_training)


def inception_reduction_a(inputs, is_training):
    # 64 x 64
    bn = tf.layers.batch_normalization(inputs, axis=3, training=is_training)

    #1x1 downsample
    conv = tf.layers.conv2d(bn, 32, kernel_size=1, strides=(1, 1), padding='same')
    conv2 = tf.layers.conv2d(conv, 32, kernel_size=3, strides=(1, 1), padding='same')
    right = tf.layers.conv2d(conv2, 32, kernel_size=3, strides=(2, 2), padding='valid')

    pooled = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    # outputs a 32 x 32

    middle = tf.layers.conv2d(bn, 32, kernel_size=3, strides=(2, 2), padding='valid')
    # outputs a 32 x 32 x 64

    combined = tf.concat([pooled, middle, right], axis=-1)

    return tf.nn.relu(combined) # 32 x 32 x 96

def inception_reduction_b(inputs, is_training):
    # 32 x 32 x 128
    bn = tf.layers.batch_normalization(inputs, axis=3, training=is_training)

    #1x1 downsample
    # conv = tf.layers.conv2d(bn, 96, kernel_size=1, strides=(1, 1), padding='same')
    conv2 = tf.layers.conv2d(bn, 96, kernel_size=3, strides=(1, 1), padding='same')
    right = tf.layers.conv2d(conv2, 96, kernel_size=3, strides=(2, 2), padding='valid')

    pooled = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    # outputs a 16 x 16

    middle = tf.layers.conv2d(bn, 96, kernel_size=3, strides=(2, 2), padding='valid')
    # outputs a 16 x 16 x 96

    combined = tf.concat([pooled, middle, right], axis=-1)

    return tf.nn.relu(combined) # 16 x 16 x 320

def inception_reduction_c(inputs, is_training):
    # 16 x 16 x 320
    bn = tf.layers.batch_normalization(inputs, axis=3, training=is_training)

    #1x1 downsample
    conv = tf.layers.conv2d(bn, 196, kernel_size=1, strides=(1, 1), padding='same')
    conv2 = tf.layers.conv2d(bn, 196, kernel_size=3, strides=(1, 1), padding='same')
    right = tf.layers.conv2d(conv2, 256, kernel_size=3, strides=(2, 2), padding='valid')

    pooled = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    # outputs a 7 x 7

    conv = tf.layers.conv2d(bn, 196, kernel_size=1, strides=(1, 1), padding='same')
    middle = tf.layers.conv2d(bn, 256, kernel_size=3, strides=(2, 2), padding='valid')
    # outputs a 7 x 7

    combined = tf.concat([pooled, middle, right], axis=-1)

    return tf.nn.relu(combined) # 7 x 7 x 832

def inception_memnet(inputs, is_training):
    # bn = tf.layers.batch_normalization(inputs, axis=3, training=is_training)
    conv = tf.layers.conv2d(inputs, 64, kernel_size=3, strides=(1, 1), padding='same')
    print(conv.get_shape().as_list())
    mem1 = resnext_block(conv, is_training, 64, num_modules=2, downsample=False)
    # reduc = inception_reduction_a(conv, is_training)
    reduc = inception_reduction_a(mem1, is_training)
    print(reduc.get_shape().as_list())

    # now 32 x 32 x 128
    mem = resnext_block(reduc, is_training, 128, num_modules=2, downsample=False)
    print(mem.get_shape().as_list())

    reduc2 = inception_reduction_b(mem, is_training)
    print(reduc2.get_shape().as_list())

    # this is now 16 x 16
    final = residual_inception(reduc2, is_training)
    print(final.get_shape().as_list())

    x = tf.nn.avg_pool(final,[1, 15, 15, 1],strides=[1,15,15,1],padding='VALID')
    print(x.get_shape().as_list())

    # now is 1 x 1 x 1 x 288
    return tf.reshape(x, [-1, 320])

def inception_memnet_v2(inputs, is_training):
    # bn = tf.layers.batch_normalization(inputs, axis=3, training=is_training)
    conv = tf.layers.conv2d(inputs, 64, kernel_size=3, strides=(1, 1), padding='same')
    print(conv.get_shape().as_list())
    mem1 = attention_module(conv, 64, is_training)
    # reduc = inception_reduction_a(conv, is_training)
    reduc = inception_reduction_a(mem1, is_training)
    print(reduc.get_shape().as_list())

    # now 32 x 32 x 128
    mem = resnext_block(reduc, is_training, 128, num_modules=2, downsample=False)
    print(mem.get_shape().as_list())

    reduc2 = inception_reduction_b(mem, is_training)
    print(reduc2.get_shape().as_list())

    # this is now 16 x 16
    final = residual_inception(reduc2, is_training)
    print(final.get_shape().as_list())

    x = tf.nn.avg_pool(final,[1, 15, 15, 1],strides=[1,15,15,1],padding='VALID')
    print(x.get_shape().as_list())

    # now is 1 x 1 x 1 x 288
    return tf.reshape(x, [-1, 320])

def inception_memnet_v3(inputs, is_training):
    # bn = tf.layers.batch_normalization(inputs, axis=3, training=is_training)
    conv = tf.layers.conv2d(inputs, 64, kernel_size=3, strides=(1, 1), padding='same')
    print(conv.get_shape().as_list())
    mem1 = attention_module(conv, 64, is_training)
    # reduc = inception_reduction_a(conv, is_training)
    reduc = inception_reduction_a(mem1, is_training)
    print(reduc.get_shape().as_list())

    # now 32 x 32 x 128
    mem = resnext_block(reduc, is_training, 128, num_modules=2, downsample=False)
    print(mem.get_shape().as_list())

    reduc2 = inception_reduction_b(mem, is_training)
    print(reduc2.get_shape().as_list())

    # this is now 16 x 16
    res_inc = residual_inception(reduc2, is_training)
    print(res_inc.get_shape().as_list())

    final = inception_reduction_c(res_inc, is_training)
    print(final.get_shape().as_list())
    x = tf.nn.avg_pool(final,[1, 7, 7, 1],strides=[1,7,7,1],padding='VALID')
    print(x.get_shape().as_list())
    return tf.reshape(x, [-1, 832])

def inception_memnet_v4(inputs, is_training):
    # bn = tf.layers.batch_normalization(inputs, axis=3, training=is_training)
    conv = tf.layers.conv2d(inputs, 64, kernel_size=3, strides=(1, 1), padding='same')
    print(conv.get_shape().as_list())
    mem1 = mini_attention_module(conv, 64, is_training)
    # reduc = inception_reduction_a(conv, is_training)
    reduc = inception_reduction_a(mem1, is_training)
    print(reduc.get_shape().as_list())

    # now 32 x 32 x 128
    mem = resnext_block(reduc, is_training, 128, num_modules=2, downsample=False)
    print(mem.get_shape().as_list())

    reduc2 = inception_reduction_b(mem, is_training)
    print(reduc2.get_shape().as_list())

    # this is now 16 x 16
    final = residual_inception(reduc2, is_training)
    print(final.get_shape().as_list())

    x = tf.nn.avg_pool(final,[1, 15, 15, 1],strides=[1,15,15,1],padding='VALID')
    print(x.get_shape().as_list())

    # now is 1 x 1 x 1 x 320
    return tf.reshape(x, [-1, 320])
