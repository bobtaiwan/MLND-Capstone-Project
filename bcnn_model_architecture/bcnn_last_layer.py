import tensorflow as tf
import numpy as np


class model:

    def __init__(self, imgs, weights=None, sess=None):
        self.images = imgs
        self.last_layer_parameters = []
        self.parameters = []
        self.cnn_layers()
        self.fc_layers()
        self.weight_file = weights

    def cnn_layers(self):

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False)
            conv = tf.nn.conv2d(self.images, weights, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False)
            conv = tf.nn.conv2d(self.conv1_1, weights, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False)
            conv = tf.nn.conv2d(self.pool1, weights, [1, 1, 1, 1], padding='SAME')

            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=False)
            conv = tf.nn.conv2d(self.conv2_1, weights, [1, 1, 1, 1], padding='SAME')

            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False)

            conv = tf.nn.conv2d(self.pool2, weights, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False)
            conv = tf.nn.conv2d(self.conv3_1, weights, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False)

            conv = tf.nn.conv2d(self.conv3_2, weights, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False)

            conv = tf.nn.conv2d(self.pool3, weights, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False)
            conv = tf.nn.conv2d(self.conv4_1, weights, [1, 1, 1, 1], padding='SAME')

            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False)
            conv = tf.nn.conv2d(self.conv4_2, weights, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False)
            conv = tf.nn.conv2d(self.pool4, weights, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False)
            conv = tf.nn.conv2d(self.conv5_1, weights, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                      stddev=1e-1), trainable=False)
            conv = tf.nn.conv2d(self.conv5_2, weights, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [weights, biases]

        print('Shape of conv5_3', self.conv5_3.get_shape())
        self.phi_I = tf.einsum('ijkm,ijkn->imn', self.conv5_3, self.conv5_3)
        print('Shape of phi_I after einsum', self.phi_I.get_shape())

        self.phi_I = tf.reshape(self.phi_I, [-1, 512 * 512])
        print('Shape of phi_I after reshape', self.phi_I.get_shape())

        self.phi_I = tf.divide(self.phi_I, 784.0)
        print('Shape of phi_I after division', self.phi_I.get_shape())

        self.y_ssqrt = tf.multiply(tf.sign(self.phi_I), tf.sqrt(tf.abs(self.phi_I) + 1e-12))
        print('Shape of y_ssqrt', self.y_ssqrt.get_shape())

        self.z_l2 = tf.nn.l2_normalize(self.y_ssqrt, axis=1)
        print('Shape of z_l2', self.z_l2.get_shape())

    def fc_layers(self):

        with tf.name_scope('fc-new') as scope:
            fc3w = tf.get_variable('weights', [512 * 512, 128], initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=True)
            fc3b = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.z_l2, fc3w), fc3b)
            self.last_layer_parameters += [fc3w, fc3b]
            self.parameters += [fc3w, fc3b]

    def load_weights(self, sess):
        weights = np.load(self.weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            removed_layer_variables = ['fc6_W', 'fc6_b', 'fc7_W', 'fc7_b', 'fc8_W', 'fc8_b']
            if not k in removed_layer_variables:
                print(k)
                print("", i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(weights[k]))
