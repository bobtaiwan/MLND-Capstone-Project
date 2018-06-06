from __future__ import print_function
import tensorflow as tf
import numpy as np
from tflearn.data_utils import shuffle
import h5py
import time
from utils.utils import random_flip_right_to_left, save_bcnn_weights, predict_test
from bcnn_model_architecture.bcnn_full_layer import model
from tqdm import tqdm

def train(sess,x_train, y_train, train_total_batch, train_batch_size):
    for i in tqdm(range(train_total_batch)):
        train_start_time = time.time()
        batch_x, batch_y = x_train[i * train_batch_size:i * train_batch_size + train_batch_size], \
                           y_train[i * train_batch_size:i * train_batch_size + train_batch_size]
        batch_x = random_flip_right_to_left(batch_x)
        sess.run([optimizer, check_op], feed_dict={imgs: batch_x, target: batch_y})
        if i % 20 == 0:
            cost = sess.run(loss, feed_dict={imgs: batch_x, target: batch_y})
            accuracy = sess.run(num_correct_preds,feed_dict={imgs: batch_x, target: batch_y})/ len(batch_x)
            cost_time = time.time() - train_start_time
            print(
                'Batch Index:{0} Training Accuracy:{1} Train Loss:{2} Time:{3}'.format(i, accuracy, cost,cost_time))

def val(sess, x_val, y_val, total_val_batch, val_batch_size):
    correct_count = 0
    cost = 0.0
    for i in range(total_val_batch):
        batch_val_x, batch_val_y = x_val[i * val_batch_size:i * val_batch_size + val_batch_size], \
                                   y_val[i * val_batch_size:i * val_batch_size + val_batch_size]
        cost += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y})
        correct_count += sess.run(num_correct_preds, feed_dict={imgs: batch_val_x, target: batch_val_y})
    val_accuracy=correct_count / len(x_val)
    print("Val Loss:{0} Accuracy:{1} ".format(cost, val_accuracy))
    return val_accuracy

def train_val(sess, x_train, y_train, x_val, y_val, epochs):
    x_train, y_train = shuffle(x_train, y_train)
    train_batch_size = 32
    val_batch_size = 64
    train_total_batch = int(len(x_train) / train_batch_size)
    total_val_batch = int(len(x_val) / val_batch_size)
    validation_accuracy_buffer = []
    for epoch in range(epochs):
        print('epoche:{0} training start'.format(epoch))
        epoch_time = time.time()
        train(sess, x_train, y_train, train_total_batch, train_batch_size)
        val_accuracy=val(sess, x_val, y_val, total_val_batch, val_batch_size)
        print('Epoch:{0} , time:{1} seconds'.format(epoch, time.time() - epoch_time))
        if epoch > 10:
            validation_accuracy_buffer.append(val_accuracy)
            if len(validation_accuracy_buffer) > 10:
                index_of_max_val_acc = np.argmax(validation_accuracy_buffer)
                if index_of_max_val_acc == 0:
                    break
                else:
                    del validation_accuracy_buffer[0]


if __name__=='__main__':
    epochs=22
    train_data = h5py.File('train_128.h5', 'r')
    val_data = h5py.File('val_128.h5', 'r')
    test_data = h5py.File('test_128.h5', 'r')
    x_train, y_train = train_data['X'], train_data['Y']
    x_val, y_val = val_data['X'], val_data['Y']
    x_test, y_test = test_data['X'], test_data['Y']
    print('Data shapes -- (train:{0}, val{1}, test{2})'.format(x_train.shape, x_val.shape, x_test.shape))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # sess = tf.Session()     ## Start session to create training graph
    imgs = tf.placeholder(tf.float32, [None, 128, 128, 3])
    target = tf.placeholder("float", [None, 128])
    bcnn = model(imgs, 'vgg16_weights.npz', sess)

    print('BCNN network created')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=bcnn.fc3l, labels=target))
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)
    check_op = tf.add_check_numerics_ops()
    correct_prediction = tf.equal(tf.argmax(bcnn.fc3l, 1), tf.argmax(target, 1))
    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    bcnn.load_initial_weights(sess)
    train_val(sess, x_train, y_train, x_val, y_val, epochs)
    save_bcnn_weights(bcnn, sess, False)
    predict_test(bcnn, sess, x_test, y_test, imgs)
