import os
import config as cfg
from sklearn import svm
import tensorflow as tf
from sklearn.externals import joblib
from tensorflow.python.ops import nn_ops
from tflearn.layers.normalization import local_response_normalization

slim = tf.contrib.slim

class Alexnet_Net :
    '''
    此类用来定义Alexnet网络及其参数，之后整体作为参数输入到Solver中
    '''
    def __init__(self, is_training=True, is_fineturn=False, is_SVM=False):
        self.image_size = cfg.Image_size
        self.batch_size = cfg.F_batch_size if is_fineturn else cfg.T_batch_size
        self.class_num = cfg.F_class_num if is_fineturn else cfg.T_class_num
        self.input_data = tf.placeholder(tf.float32,[None, self.image_size, self.image_size,3], name='input')
        self.logits = self.build_network(self.input_data, self.class_num, is_svm=is_SVM, is_training=is_training)

        if is_training == True :
            self.label = tf.placeholder(tf.float32, [None, self.class_num], name='label')
            self.loss_layer(self.logits, self.label)
            self.accuracy = self.get_accuracy(self.logits, self.label)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self, input, output, is_svm= False, scope='R-CNN',is_training=True, keep_prob=0.5):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                activation_fn=nn_ops.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net = slim.conv2d(input, 96, 11, stride=4, scope='conv_1')
                net = slim.max_pool2d(net, 3, stride=2, scope='pool_2')
                net = local_response_normalization(net)
                net = slim.conv2d(net, 256, 5, scope='conv_3')
                net = slim.max_pool2d(net, 3, stride=2, scope='pool_2')
                net = local_response_normalization(net)
                net = slim.conv2d(net, 384, 3, scope='conv_4')
                net = slim.conv2d(net, 384, 3, scope='conv_5')
                net = slim.conv2d(net, 256, 3, scope='conv_6')
                net = slim.max_pool2d(net, 3, stride=2, scope='pool_7')
                net = local_response_normalization(net)
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 4096, activation_fn=self.tanh(), scope='fc_8')
                net = slim.dropout(net, keep_prob=keep_prob,is_training=is_training, scope='dropout9')
                net = slim.fully_connected(net, 4096, activation_fn=self.tanh(), scope='fc_10')
                if is_svm:
                    return net
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout11')
                net = slim.fully_connected(net, output, activation_fn=self.softmax(), scope='fc_11')
        return net

    def loss_layer(self, y_pred, y_true):
        with tf.name_scope("Crossentropy"):
            y_pred = tf.clip_by_value(y_pred, tf.cast(1e-10, dtype=tf.float32),tf.cast(1. - 1e-10, dtype=tf.float32))
            cross_entropy = - tf.reduce_sum(y_true * tf.log(y_pred),reduction_indices=len(y_pred.get_shape()) - 1)
            loss = tf.reduce_mean(cross_entropy)
            tf.losses.add_loss(loss)
            tf.summary.scalar('loss', loss)

    def get_accuracy(self, y_pred, y_true):
        y_pred_maxs =(tf.argmax(y_pred,1))
        y_true_maxs =(tf.argmax(y_true,1))
        num = tf.count_nonzero((y_true_maxs-y_pred_maxs))
        result = 1-(num/self.batch_size)
        return result
    def softmax(self):
        def op(inputs):
            return tf.nn.softmax(inputs)
        return op
    def tanh(self):
        def op(inputs):
            return tf.tanh(inputs)
        return op

class SVM :
    def __init__(self, data):
        self.data = data
        self.data_save_path = cfg.SVM_and_Reg_save
        self.output = cfg.Out_put
    def train(self):
        svms=[]
        data_dirs = os.listdir(self.data_save_path)
        for data_dir in data_dirs:
            images, labels = self.data.get_SVM_data(data_dir)
            clf = svm.LinearSVC()
            clf.fit(images, labels)
            svms.append(clf)
            SVM_model_path = os.path.join(self.output, 'SVM_model')
            if not os.path.exists(SVM_model_path):
                os.makedirs(SVM_model_path)
            joblib.dump(clf, os.path.join(SVM_model_path,  str(data_dir)+ '_svm.pkl'))

class Reg_Net(object):
    def __init__(self, is_training=True):
        self.output_num = cfg.R_class_num
        self.input_data = tf.placeholder(tf.float32, [None, 4096], name='input')
        self.logits = self.build_network(self.input_data, self.output_num, is_training=is_training)
        if is_training:
            self.label = tf.placeholder(tf.float32, [None, self.output_num], name='input')
            self.loss_layer(self.logits, self.label)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self, input_image, output_num, is_training= True, scope='regression_box', keep_prob=0.5):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=self.tanh(),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net = slim.fully_connected(input_image, 4096, scope='fc_1')
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout11')
                net = slim.fully_connected(net, output_num, scope='fc_2')
                return net

    def loss_layer(self,y_pred, y_true):
        no_object_loss = tf.reduce_mean(tf.square((1 - y_true[:, 0]) * y_pred[:, 0]))
        object_loss = tf.reduce_mean(tf.square((y_true[:, 0]) * (y_pred[:, 0] - 1)))

        loss = (tf.reduce_mean(y_true[:, 0] * (
                 tf.reduce_sum(tf.square(y_true[:, 1:5] - y_pred[:, 1:5]), 1))) + no_object_loss + object_loss)
        tf.losses.add_loss(loss)
        tf.summary.scalar('loss', loss)

    def tanh(self):
        def op(inputs):
            return tf.tanh(inputs)
        return op





