import random
import tensorflow as tf
import numpy as np

def random_flip_right_to_left(image_batch):
    result = []
    for n in range(image_batch.shape[0]):
        if bool(random.getrandbits(1)):
            result.append(image_batch[n][:, ::-1, :])
        else:
            result.append(image_batch[n])
    return result


def save_bcnn_weights(bcnn, sess,is_only_save_last_layer):

    if is_only_save_last_layer:
        last_layer_weights = []
        for v in bcnn.parameters:
            print(v)
            if v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                last_layer_weights.append(sess.run(v))
        np.savez('last_layers_epoch_128.npz', last_layer_weights)
        print("Last layer weights saved")
    else:
        full_layer_weights = []
        for v in bcnn.parameters:
            print(v)
            if v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                print('Printing Trainable Variables :', sess.run(v).shape)
                full_layer_weights.append(sess.run(v))
        for v in bcnn.last_layer_parameters:
            print(v)
            if v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                full_layer_weights.append(sess.run(v))
        np.savez('full_layers_epoch_128.npz', full_layer_weights)
        print("full layer weights saved")


def predict_test(bcnn, sess, x_test, y_test, imgs):
    prediction = tf.argmax(bcnn.fc3l, 1)

    total_test_count = len(x_test)
    correct_test_count = 0
    test_batch_size = 128
    total_test_batch = int(total_test_count / test_batch_size)
    with open('pred.txt', 'w') as text_file:
        for i in range(total_test_batch):
            batch_test_x, x_id = x_test[i * test_batch_size:i * test_batch_size + test_batch_size], \
                                 y_test[i * test_batch_size:i * test_batch_size + test_batch_size]
            pred = sess.run(prediction, feed_dict={imgs: batch_test_x})
            for i in range(len(x_id)):
                text_file.write('{0},{1}\n'.format(int(x_id[i]), pred[i] + 1))

    print('Prediction completed')