# setup variables
# Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
# bconv1 = tf.get_variable("bconv1", shape=[32])
#
# Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
# bconv2 = tf.get_variable("bconv2", shape=[32])
#
# Wconv3 = tf.get_variable("Wconv3", shape=[5, 5, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
# bconv3 = tf.get_variable("bconv3", shape=[32])
#
# # here, X is 64 x 64 x 3
#
# # define our graph (e.g. two_layer_convnet)
# a1 = tf.nn.conv2d(self.X, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + bconv1
# h1 = tf.nn.relu(a1)
# sb1 = tf.layers.batch_normalization(h1, axis=3, training=self.is_training)
# # 2x2 max pooling, stride 2
# p1 = tf.nn.max_pool(sb1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# p1 = tf.nn.dropout(p1, self.keep_prob)
# # we now have N x 32 x 32 x 32
# a2 = tf.nn.conv2d(p1, Wconv2, strides=[1, 1, 1, 1], padding='SAME') + bconv2
# h2 = tf.nn.relu(a2)
# sb2 = tf.layers.batch_normalization(h2, axis=3, training=self.is_training)
#
# p2 = tf.nn.max_pool(sb2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# # we now have N x 16 x 16 x 32
# p2 = tf.nn.dropout(p2, self.keep_prob)
# a3 = tf.nn.conv2d(p2, Wconv3, strides=[1, 1, 1, 1], padding='SAME') + bconv3
# h3 = tf.nn.relu(a3)
# sb3 = tf.layers.batch_normalization(h3, axis=3, training=self.is_training)
# p3 = tf.nn.max_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# p3 = tf.nn.dropout(p3, self.keep_prob)
