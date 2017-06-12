from data_utils import load_tiny_imagenet
import tensorflow as tf
import numpy as np
import math
from basic_module import basic_module
from attention_module import attention_module
from maxout import maxout
from attention.resnext import *
from attention.inception import inception_memnet, inception_memnet_v2, inception_memnet_v3, inception_memnet_v4

class ImageModel:
    def __init__(self):
        self.add_placeholders()
        self.y_out = self.add_prediction_op()
        self.loss = self.add_loss_op(self.y_out)
        self.training_op = self.add_optimization_op(self.loss)

        # tensorboard reporting
        self.writer = tf.summary.FileWriter('logs/v_simple')
        self.add_summary_op()

    def add_summary_op(self):
        self.train_acc_placeholder = tf.placeholder(tf.float32, name='train_acc')
        self.train_loss_placeholder = tf.placeholder(tf.float32, name='train_loss')
        self.valid_acc_placeholder = tf.placeholder(tf.float32, name='valid_acc')
        self.valid_loss_placeholder = tf.placeholder(tf.float32, name='valid_loss')

        train_acc_sum = tf.summary.scalar("train_accuracy", self.train_acc_placeholder)
        train_loss_sum = tf.summary.scalar("train_loss", self.train_loss_placeholder)
        valid_acc_sum = tf.summary.scalar("validation_accuracy", self.valid_acc_placeholder)
        valid_loss_sum = tf.summary.scalar("validation_loss", self.valid_loss_placeholder)
        self.merged_info = tf.summary.merge_all()

    def record_summary(self, session, t_loss, t_acc, v_loss, v_acc, epoch):
        feed_dict = {}
        feed_dict[self.train_acc_placeholder] = t_acc
        feed_dict[self.train_loss_placeholder] = t_loss
        feed_dict[self.valid_acc_placeholder] = v_acc
        feed_dict[self.valid_loss_placeholder] = v_loss

        summary = session.run([self.merged_info], feed_dict)
        self.writer.add_summary(summary[0], epoch)

    def add_placeholders(self):
        self.X = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.y = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self.iteration = tf.placeholder(tf.int32)

    def add_prediction_op(self):
        # x = resnext_model(self.X, self.is_training)
        x = inception_memnet(self.X, self.is_training)
        # x = inception_memnet_v2(self.X, self.is_training)
        # x = inception_memnet_v4(self.X, self.is_training)
        # a_flat = tf.reshape(x,[-1, 512])
        dropout = tf.layers.dropout(inputs=x, rate=0.3, training=self.is_training)
        # y_out = tf.layers.dense(inputs=dropout, units=200)
        hidden = tf.layers.dense(inputs=dropout, units=1000)
        # dropout = tf.layers.dropout(inputs=hidden, rate=0.2, training=self.is_training)
        y_out = tf.layers.dense(inputs=hidden, units=200)
        return y_out

    def add_loss_op(self, y_out):
        one_hot_labels = tf.one_hot(self.y, depth=200)
        mean_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=y_out))
        # self.loss = mean_loss
        return mean_loss

    def add_optimization_op(self, loss):
        # lr = tf.train.exponential_decay(1e-3, self.iteration, 1000, 0.96, staircase=True)
        # lr = tf.train.exponential_decay(1e-3, self.iteration, 1000, 0.96, staircase=True)
        lr = 3e-5
        optimizer = tf.train.AdamOptimizer(lr) # select optimizer and set learning rate
        # batch normalization in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = optimizer.minimize(loss)
            return train_step

    def batch_validation(self, session, Xd, yd, batch_size=32):
        correct_prediction = tf.equal(tf.argmax(self.y_out,1), self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_indicies = np.arange(Xd.shape[0])
        variables = [self.loss, correct_prediction, accuracy]
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
                         self.is_training: False,
                         self.keep_prob: 1.0}
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("VALIDATION: Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct))
        return total_loss, total_correct

    def run_with_valid(self, session, Xd, yd, Xv, yv, epochs=1, batch_size=32, print_every=200):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(self.y_out,1), self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # training_now = training is not None

        # setup the saver object
        # saver = tf.train.Saver(tf.trainable_variables())
        saver = tf.train.Saver(max_to_keep=epochs)

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self.loss, correct_prediction, accuracy]
        training_vars = [self.loss, correct_prediction, self.training_op]

        # counter
        iter_cnt = 0
        for e in range(11, epochs):
            # shuffle indicies
            train_indicies = np.arange(Xd.shape[0])
            np.random.shuffle(train_indicies)

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
                             self.is_training: True,
                             self.keep_prob: 0.7,
                             self.iteration: iter_cnt}
                # get batch size
                actual_batch_size = yd[i:i+batch_size].shape[0]

                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(training_vars,feed_dict=feed_dict)

                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)

                # print every now and then
                if (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                          .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
            saver.save(session, "ckpts/v_simple/model.ckpt", global_step=e)
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,e+1))
            # compute the validation loss and accuracy
            valid_loss, valid_acc = self.batch_validation(session, Xv, yv)
            self.record_summary(session, total_loss, total_correct, valid_loss, valid_acc, e+1)


        return total_loss,total_correct

    def compute_test_labels(self, session, Xd, test_files, batch_size=32):
        predictions = tf.argmax(self.y_out,1)
        train_indicies = np.arange(Xd.shape[0])
        result = []

        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]

            # create a feed dictionary for this batch
            feed_dict = {self.X: Xd[idx,:],
                         self.is_training: False}
            variables = [predictions]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            preds = session.run(variables,feed_dict=feed_dict)
            # print(preds[0].shape)
            result.extend(preds[0].tolist())

        # check the result length
        print(str(len(result)) + ": should be 10,000")
        with open('data/wnids.txt') as wnids_file:
            wnids = wnids_file.readlines()
            with open('tediris.txt', 'w') as outfile:
                for idx, pred in enumerate(result):
                    outfile.write(test_files[idx] + " " + wnids[pred] + "")

    def run(self, session, Xd, yd, training, epochs=1, batch_size=32, print_every=200):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(self.y_out,1), self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self.loss, correct_prediction, accuracy]
        if training:
            variables[-1] = self.training_op

        keep_prob = 1.0
        if (training):
            keep_prob = 0.7

        # counter
        iter_cnt = 0
        for e in range(epochs):
            # shuffle indicies
            train_indicies = np.arange(Xd.shape[0])
            np.random.shuffle(train_indicies)

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
                             self.is_training: training,
                             self.keep_prob: keep_prob}
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

            # save the current model
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
    test_files = data['test_files']
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
    cp_saver = tf.train.Saver(tf.trainable_variables())
    # cp_saver = tf.train.Saver()
    cp_saver.restore(sess, "ckpts/v_simple/model.ckpt-10")
    # print('Training')
    # # logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    # # train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
    # model.run(sess, X_train, y_train, True, epochs=10)
    # print('Validation')
    # model.run(sess, X_val, y_val, False, epochs=1)
    # # run_model(sess,X_val,y_val)
    model.run_with_valid(sess, X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
    # print(model.batch_validation(sess, X_val, y_val))
    # model.compute_test_labels(sess, X_test, test_files)


if __name__ == "__main__":
    main()
