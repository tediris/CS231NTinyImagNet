from data_utils import load_tiny_imagenet
import tensorflow as tf
import numpy as np
import math
from basic_module import basic_module
from attention_module import attention_module
from maxout import maxout
from squeezenet import SqueezeNet

class ImageModel:
    def __init__(self, session):
        self.session = session
        self.add_placeholders()
        self.squeezenet = SqueezeNet(save_path='squeezenet/squeezenet.ckpt', sess=session)
        self.y_out = self.add_prediction_op()
        self.loss = self.add_loss_op(self.y_out)
        self.training_op = self.add_optimization_op(self.loss)
        self.correct_prediction, self.accuracy = self.add_accuracy_op()

    def add_placeholders(self):
        self.X = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.y = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

    def add_prediction_op(self):
        x = self.X
        filter_depths = [(3, 32), (32, 32), (32, 32)]
        for i in range(3):
            name = "layer" + str(i)
            # x = basic_module(x, self.keep_prob, self.is_training, filter_depths[i][0], filter_depths[i][1], name)
            x = attention_module(x, self.keep_prob, self.is_training, filter_depths[i][0], filter_depths[i][1], name)
        # we now have a N x 8 x 8 x 32
        # do a couple dense layers
        # a_flat = tf.reshape(p3,[-1, 8 * 8 * 32])
        a_flat = tf.reshape(x,[-1, 8 * 8 * 32])

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

    def add_accuracy_op(self):
        correct_prediction = tf.equal(tf.argmax(self.y_out,1), self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return correct_prediction, accuracy

    def run_batch(self, session, variables, X, y, is_training=False, keep_prob=1.0):
        # create a feed dictionary for this batch
        feed_dict = {self.X: X,
                     self.y: y,
                     self.is_training: is_training,
                     self.keep_prob: keep_prob}
        return session.run(variables,feed_dict=feed_dict)

    def batch_validation(self, session, Xd, yd, batch_size=32):
        train_indicies = np.arange(Xd.shape[0])
        variables = [self.loss, self.correct_prediction, self.accuracy]
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]

            # create a feed dictionary for this batch
            # feed_dict = {self.X: Xd[idx,:],
            #              self.y: yd[idx],
            #              self.is_training: False,
            #              self.keep_prob: 1.0}
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            # loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            loss, corr, _ = self.run_batch(session, variables, Xd[idx,:], yd[idx], False, 1.0)

            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("VALIDATION: Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct))

    def run_with_valid(self, session, Xd, yd, Xv, yv, epochs=1, batch_size=32, print_every=200):
        # have tensorflow compute accuracy

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        # training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        # variables = [self.loss, self.correct_prediction, self.accuracy]
        training_vars = [self.loss, self.correct_prediction, self.training_op]

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
                # feed_dict = {self.X: Xd[idx,:],
                #              self.y: yd[idx],
                #              self.is_training: True,
                #              self.keep_prob: 0.5}
                # get batch size
                actual_batch_size = yd[i:i+batch_size].shape[0]

                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                # loss, corr, _ = session.run(training_vars,feed_dict=feed_dict)
                loss, corr, _ = self.run_batch(session, training_vars, Xd[idx,:], yd[idx], False, 1.0)

                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)

                # print every now and then
                if (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                          .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,e+1))
            # compute the validation loss and accuracy
            self.batch_validation(session, Xv, yv, )


        return total_loss,total_correct

def main():
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
    print('Building model...')
    model = ImageModel(sess)
    sess.run(tf.global_variables_initializer())
    # print('Training')
    # # logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    # # train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
    # model.run(sess, X_train, y_train, True, epochs=10)
    # print('Validation')
    # model.run(sess, X_val, y_val, False, epochs=1)
    # # run_model(sess,X_val,y_val)
    model.run_with_valid(sess, X_train, y_train, X_val, y_val, epochs=10)


if __name__ == "__main__":
    main()
