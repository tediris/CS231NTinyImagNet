from data_utils import load_tiny_imagenet
import tensorflow as tf
import numpy as np
import math

class ImageModel:
    def __init__(self):
        self.add_placeholders()
        self.y_out = self.add_prediction_op()
        self.loss = self.add_loss_op(self.y_out)
        self.training_op = self.add_optimization_op(self.loss)

    def add_placeholders(self):
        self.X = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.y = tf.placeholder(tf.int64, [None])
        self.dropout_prob = tf.placeholder(tf.bool)
        self.is_training = tf.placeholder(tf.bool)

    def add_prediction_op(self):
        # setup variables
        Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
        bconv1 = tf.get_variable("bconv1", shape=[32])

        Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
        bconv2 = tf.get_variable("bconv2", shape=[32])

        Wconv3 = tf.get_variable("Wconv3", shape=[5, 5, 32, 32], initializer = tf.contrib.layers.xavier_initializer())
        bconv3 = tf.get_variable("bconv3", shape=[32])

        # here, X is 64 x 64 x 3

        # define our graph (e.g. two_layer_convnet)
        a1 = tf.nn.conv2d(self.X, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + bconv1
        h1 = tf.nn.relu(a1)
        sb1 = tf.layers.batch_normalization(h1, axis=3, training=self.is_training)
        # 2x2 max pooling, stride 2
        p1 = tf.nn.max_pool(sb1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # we now have N x 32 x 32 x 32
        a2 = tf.nn.conv2d(p1, Wconv2, strides=[1, 1, 1, 1], padding='SAME') + bconv2
        h2 = tf.nn.relu(a2)
        sb2 = tf.layers.batch_normalization(h2, axis=3, training=self.is_training)

        p2 = tf.nn.max_pool(sb2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # we now have N x 16 x 16 x 32
        a3 = tf.nn.conv2d(p2, Wconv3, strides=[1, 1, 1, 1], padding='SAME') + bconv3
        h3 = tf.nn.relu(a3)
        sb3 = tf.layers.batch_normalization(h3, axis=3, training=self.is_training)
        p3 = tf.nn.max_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # we now have a N x 8 x 8 x 32
        # do a couple dense layers
        a_flat = tf.reshape(p3,[-1, 8 * 8 * 32])

        dense = tf.layers.dense(inputs=a_flat, units=2048, activation=tf.nn.relu)
        y_out = tf.layers.dense(inputs=dense, units=200)
        return y_out

    def add_loss_op(self, y_out):
        one_hot_labels = tf.one_hot(self.y, depth=200)
        mean_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=y_out))
        # self.loss = mean_loss
        return mean_loss

    def add_optimization_op(self, loss):
        optimizer = tf.train.AdamOptimizer() # select optimizer and set learning rate
        # batch normalization in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = optimizer.minimize(loss)
            return train_step

    def run(self, session, Xd, yd, training, epochs=1, batch_size=32, print_every=200):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(self.y_out,1), self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        # training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self.loss, correct_prediction, accuracy]
        if training:
            variables[-1] = self.training_op

        # counter
        iter_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                # generate indicies for the batch
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+batch_size]

                # create a feed dictionary for this batch
                feed_dict = {self.X: Xd[idx,:],
                             self.y: yd[idx],
                             self.is_training: training}
                # get batch size
                actual_batch_size = yd[i:i+batch_size].shape[0]

                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables,feed_dict=feed_dict)

                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)

                # print every now and then
                if training and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                          .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,e+1))
            # if plot_losses:
            #     plt.plot(losses)
            #     plt.grid(True)
            #     plt.title('Epoch {} Loss'.format(e+1))
            #     plt.xlabel('minibatch number')
            #     plt.ylabel('minibatch loss')
            #     plt.show()
        return total_loss,total_correct

def main():
    print('Building model...')
    model = ImageModel()
    print('Loading ImageNet...')
    data = load_tiny_imagenet("./data")
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    class_names = data['class_names']
    # y_test = data['y_test']

    # print('Test labels shape: ', y_test.shape)
    # permute the data axes
    X_train = np.transpose(X_train, (0, 2, 3, 1))
    X_val = np.transpose(X_val, (0, 2, 3, 1))
    X_test = np.transpose(X_test, (0, 2, 3, 1))

    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print(len(class_names))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print('Training')
    # logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    # train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
    model.run(sess, X_train, y_train, True)
    # print('Validation')
    # run_model(sess,X_val,y_val)


if __name__ == "__main__":
    main()
