import tensorflow as tf
from keras import Model
from keras.layers import Input, Conv2D, Conv1D, BatchNormalization, Activation, Conv2DTranspose, Lambda, Reshape, ELU
import keras.backend as K


class ATAB(Model):
    def __init__(self, filter, v_filter, kernel, dilation_rate):
        super(ATAB, self).__init__()
        self.filter = filter
        self.v_filter = v_filter
        self.kernel = kernel
        self.dilation_rate = dilation_rate
        self.a1_2 = Activation('sigmoid')
        self.d1_q = Conv2D(filters=self.filter, kernel_size=self.kernel, strides=1,
                           dilation_rate=self.dilation_rate, data_format='channels_last', padding='same')
        self.d1_k = Conv2D(filters=self.filter, kernel_size=self.kernel, strides=1,
                           dilation_rate=self.dilation_rate, data_format='channels_last', padding='same')
        self.d1_v = Conv2D(filters=self.v_filter, kernel_size=self.kernel, strides=1,
                           dilation_rate=self.dilation_rate, data_format='channels_last', padding='same')

        self.QK = tf.matmul#矩阵的乘法
        self.distribution = tf.nn.softmax
        self.QKV = tf.matmul
        self.product_1 = tf.multiply

    def call(self, X):# the shape of X is  (bz, 249, 256, channels)
        # X = tf.expand_dims(X, axis=-1)
        x1_q = self.d1_q(X)
        x1_k = self.d1_k(X)
        x1_v = self.d1_v(X)
        scores = self.QK(x1_q, x1_k, transpose_b=True)
        distribution = self.distribution(scores)
        Value = self.QKV(distribution, x1_v)
        # ax1_2 = self.a1_2(Value)
        # outcome_l1 = self.product_1(ax1_2, X)
        return Value

class AFAB(Model):
    def __init__(self, filter, v_filter,  kernel, dilation_rate):
        super(AFAB, self).__init__()
        self.filter = filter
        self.v_filter = v_filter
        self.dilation_rate = dilation_rate
        self.kernel = kernel
        self.a1_2 = Activation('sigmoid')

        self.d1_q = Conv2D(filters=self.filter, kernel_size=self.kernel, strides=1,
                           dilation_rate=self.dilation_rate, data_format='channels_last', padding='same')
        self.d1_k = Conv2D(filters=self.filter, kernel_size=self.kernel, strides=1,
                           dilation_rate=self.dilation_rate, data_format='channels_last', padding='same')
        self.d1_v = Conv2D(filters=self.v_filter, kernel_size=self.kernel, strides=1,
                           dilation_rate=self.dilation_rate, data_format='channels_last', padding='same')

        self.QK = tf.matmul#矩阵的乘法
        self.distribution = tf.nn.softmax
        self.QKV = tf.matmul
        self.product_1 = tf.multiply

    def call(self, X):# the shape of X is  (bz, 249, 256, channels)
        # X = tf.expand_dims(X, axis=-1)
        X = tf.transpose(X, perm=[0, 2, 1, 3])
        # X = Reshape((X.shape[2], X.shape[1], X.shape[-1]))(X)
        x1_q = self.d1_q(X)
        x1_k = self.d1_k(X)
        x1_v = self.d1_v(X)
        scores = self.QK(x1_q, x1_k, transpose_b=True)
        distribution = self.distribution(scores)
        Value = self.QKV(distribution, x1_v)
        # ax1_2 = self.a1_2(Value)
        # outcome_l1 = self.product_1(ax1_2, X)
        return Value

class ATFA(Model):
    def __init__(self, concatenation, kernel, dilation, filter_AFAB, filter_ATAB, v_AFAB, v_ATAB):
        super(ATFA, self).__init__()
        # self.alpha = alpha
        # self.beta = beta

        """
        Args:
        kernel: the kernel size of AFAB and ATAB, in terms of their conv2d
        dilation: the dilation rate of AFAB and ATAB, in terms of their conv2d
        filter_AFAB: the num of fitlers/channels of AFAB, in terms of their conv2d
        filter_ATAB: the num of fitlers/channels of ATAB, in terms of their conv2d
        v_AFAB: the num of fitlers/channels of conv2d for value in AFAB, in terms of their conv2d
        v_ATAB: the num of fitlers/channels of conv2d for value in ATAB, in terms of their conv2d
        """
        self.concatenation = concatenation
        self.kernel = kernel
        self.dilation = dilation
        self.filter_AFAB = filter_AFAB
        self.filter_ATAB = filter_ATAB
        self.v_AFAB = v_AFAB
        self.v_ATAB = v_ATAB
        self.att_freq = AFAB(filter=self.filter_AFAB, v_filter= self.v_AFAB, kernel= self.kernel, dilation_rate= self.dilation)
        self.att_time = ATAB(filter=self.filter_ATAB, v_filter= self.v_ATAB, kernel= self.kernel, dilation_rate= self.dilation)
        self.conv2d_final = Conv2D(filters=64, strides=1, kernel_size=self.kernel, padding='same')

        # self.alpha = self.add_weight(
        #     shape=None,
        #     initializer='random_normal',
        #     trainable=True
        # )
        # self.beta = self.add_weight(
        #     shape=None,
        #     initializer='random_normal',
        #     trainable=True
        # )

    def build(self, input_shape):
        # 创建一个可训练的权重变量矩阵
        self.alpha = self.add_weight(name='kernel_alpha',
                                      # shape=(input_shape[0], input_shape[1], input_shape[2], self.v_AFAB),  # 输入tensor要算上batch的维度
                                     shape=(2, 249, 64, self.v_AFAB),
                                      initializer='uniform',
                                      trainable=True)  # 如果要定义可训练参数这里一定要选择True
        self.beta = self.add_weight(name='kernel_beta',
                                    # shape=(input_shape[0], input_shape[1], input_shape[2], self.v_ATAB), # 输入tensor要算上batch的维度
                                     shape=(2, 249, 64, self.v_ATAB),  # 输入tensor要算上batch的维度
                                     initializer='uniform',
                                     trainable=True)  # 如果要定义可训练参数这里一定要选择True
        super(ATFA, self).build(input_shape)  # 这行代码一定要加上，super主要是调用AIA的父类（Layer）的build方法。

    def call(self, inputs, training=None, mask=None):# the shape of inputs is  (bz, 249, 256)
        out_freq = self.att_freq(inputs)
        out_time = self.att_time(inputs)
        # inputs = tf.expand_dims(inputs, axis=-1)
        # out_freq_shape = out_freq.get_shape()
        # out_freq = Reshape((out_freq.shape[2], out_freq.shape[1], out_freq.shape[-1]))(out_freq)
        out_freq = tf.transpose(out_freq, perm=[0, 2, 1, 3])
        if self.concatenation:
            final_res = tf.concat([tf.multiply(self.alpha, out_freq), tf.multiply(self.beta, out_time), inputs],
                                  axis=-1)
            final_res = self.conv2d_final(final_res)
        else:
            final_res = tf.multiply(self.alpha, out_freq) + tf.multiply(self.beta, out_time) + inputs
        return final_res

# Afab = AFAB(filter=16, v_filter= 64, kernel=1, dilation_rate=1)
# Atab = ATAB(filter=16, v_filter= 64, kernel=1, dilation_rate=1)
model = ATFA(kernel=5, dilation=1, filter_AFAB=16, filter_ATAB=16, v_AFAB=64, v_ATAB=64, concatenation=True)

# input_darwin = K.random_uniform(shape= [2, 126, 16, 64], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res = model(input_darwin)
# print(res.shape)
