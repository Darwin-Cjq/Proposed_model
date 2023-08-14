import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, AvgPool2D, GlobalAvgPool2D, Lambda
import keras.backend as K


class sub_layer(Model):
    def __init__(self, pooling_size, filters):
        super(sub_layer, self).__init__()
        self.pooling_size = pooling_size
        self.filters = filters
        self.avg_pool = AvgPool2D(pool_size=self.pooling_size, strides=1, padding='same',data_format='channels_last')
        self.conv1_1 = Conv2D(filters=self.filters, kernel_size=1, strides=1, padding='same', data_format='channels_last')
        self.global_avg = GlobalAvgPool2D

    def call(self, inputs, training=None, mask=None):
        x_pool = self.avg_pool(inputs)
        # x = tf.expand_dims(x_pool, axis=-1)
        x = self.conv1_1(x_pool)
        # the output shape is about (bz, time, freq, 1)
        return x

class AHA(Model):
    def __init__(self, num_blocks, blocks_container, pooling_size, sub_filters):
        super(AHA, self).__init__()
        self.num_blocks = num_blocks
        self.blocks_container = blocks_container
        self.pooling_size = pooling_size
        self.sub_filters = sub_filters
        for i in range(self.num_blocks):
            self.blocks_container.append(sub_layer(pooling_size=self.pooling_size, filters=self.sub_filters))
        self.SoftMax = tf.nn.softmax
        # self.gamma = self.add_weight(name='kernel_alpha',
        #                              shape=None,
        #                              # 输入tensor要算上batch的维度
        #                              initializer='uniform',
        #                              trainable=True)
    def build(self, input_shape):
        # 创建一个可训练的权重变量矩阵
        self.gamma = self.add_weight(name='kernel_alpha',
                                      # shape=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]),  # 输入tensor要算上batch的维度
                                     shape=(2, 249, 64, 64),
                                      initializer='uniform',
                                      trainable=True)  # 如果要定义可训练参数这里一定要选择True

        super(AHA, self).build(input_shape)  # 这行代码一定要加上，super主要是调用AIA的父类（Layer）的build方法。

    def call(self, inputs):#the input shape is about (bz, time, freq, channel, num_blocks)
        x_blocks = Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': self.num_blocks})(inputs)
        x_0 = self.blocks_container[0](tf.squeeze(x_blocks[0], axis=-1))
        for i in range(1, self.num_blocks):
            temp = self.blocks_container[i](tf.squeeze(x_blocks[i], axis=-1))
            x_0 = tf.concat([x_0, temp], axis=-1)
        x = tf.expand_dims(x_0, axis=-1)
        w = self.SoftMax(x, axis=-2)
        wx = tf.matmul(inputs, w)
        wx = tf.squeeze(wx, axis=-1)
        out = tf.multiply(self.gamma, wx) + tf.squeeze(x_blocks[-1], axis=-1)
        return out

model = AHA(num_blocks=3, blocks_container=[], pooling_size= 2, sub_filters=1)
# input_darwin = K.random_uniform(shape= [2, 126, 16, 4, 3], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res = model_param(input_darwin)
# print(res.shape)
