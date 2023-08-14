"""
the baseline is encoder and decoder

"encoder" contains 5 blocks, each block comprise  one conv2d module + batch normalizaiton + activation function
"middle layer" is the GLU
"decoder contains" 5 blocks, each block comprise one deconv2d module + batch normalizaiton + activation function
"""
# import librosa
# from scipy.io.wavfile import write
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Conv2D, Conv1D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D, Conv2DTranspose, Lambda, Reshape, ELU
import tensorflow.keras.backend as K
import os
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.keras
# import evaluation



#define the model
class gate_dilation(Model):
    def __init__(self, filter, kernel_size_l1, kernel_size_l2, dilation_rate, data_format, padding):
        super(gate_dilation, self).__init__()
        self.filter = filter
        self.kernel_size_l1 = kernel_size_l1
        self.kernel_size_l2 = kernel_size_l2
        self.dilation_rate = dilation_rate
        self.data_format = data_format
        self.padding = padding
        self.a1_1 = ELU(alpha=1.0)
        self.a1_2 = Activation('sigmoid')

        #complex diconv2d
        self.d1_1 = Conv1D(filters=self.filter, kernel_size=self.kernel_size_l1, strides=1,
                           dilation_rate=self.dilation_rate, data_format=self.data_format, padding= self.padding)

        #complex diconv2d
        self.d1_2 = Conv1D(filters=self.filter, kernel_size=self.kernel_size_l1, strides=1,
                           dilation_rate=self.dilation_rate, data_format=self.data_format, padding= self.padding)
        self.product_1 = tf.multiply
        self.d2_1 = Conv1D(filters=self.filter, kernel_size=self.kernel_size_l2, strides=1,
                           dilation_rate=self.dilation_rate, data_format=self.data_format, padding= self.padding)
        self.d2_2 = Conv1D(filters=self.filter, kernel_size=self.kernel_size_l2, strides=1,
                           dilation_rate=self.dilation_rate, data_format=self.data_format, padding= self.padding)
        self.b2_1 = BatchNormalization()
        self.b2_2 = BatchNormalization()
        self.a2_1 = tf.keras.layers.LeakyReLU()
        self.a2_2 = tf.keras.layers.LeakyReLU()
        self.product_2 = tf.add
        # self.product_3 = tf.concat
    def call(self, inputs):
        x1_1 = self.d1_1(inputs)
        x1_2 = self.d1_2(inputs)
        ax1_1 = self.a1_1(x1_1)
        ax1_2 = self.a1_2(x1_2)
        outcome_l1 = self.product_1(ax1_2, ax1_1)
        x2_1 = self.d2_1(outcome_l1)
        x2_2 = self.d2_2(outcome_l1)
        b2_1 = self.b2_1(x2_1)
        b2_2 = self.b2_2(x2_2)
        jump_out = self.a2_1(b2_2)
        resi_out = self.product_2(inputs, b2_1)
        resi_out = self.a2_2(resi_out)
        # resi_jump = self.product_3([resi_out, jump_out], axis=-1)
        return  resi_out, jump_out


class Project_1129(Model):
    def __init__(self, feature_timeaxis, gate_filter, kernel_size, strides, gate_kernel_1, gate_kernel_2):
        super(Project_1129, self).__init__()
        self.feature_timeaxis = feature_timeaxis
        self.gate_filter = gate_filter
        self.kernel_size = kernel_size
        self.gate_kernel_1 = gate_kernel_1
        self.gate_kernel_2 = gate_kernel_2
        self.strides = strides
        self.inp_reshape = tf.expand_dims

        self.c1 = Conv2D(filters=4, kernel_size=self.kernel_size, strides=self.strides, padding= 'same',
                                     data_format= 'channels_last')
        self.b1 = BatchNormalization()
        self.a1 = ELU(alpha=1.0)

        self.c2 = Conv2D(filters=8, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                     data_format='channels_last')
        self.b2 = BatchNormalization()
        self.a2 = ELU(alpha=1.0)

        self.c3 = Conv2D(filters=16, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                     data_format='channels_last')
        self.b3 = BatchNormalization()
        self.a3 = ELU(alpha=1.0)

        self.c4 = Conv2D(filters=32, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                     data_format='channels_last')
        self.b4 = BatchNormalization()
        self.a4 = ELU(alpha=1.0)

        self.c5 = Conv2D(filters=64, kernel_size=self.kernel_size, strides=self.strides, padding= 'same',
                                     data_format= 'channels_last')
        self.b5 = BatchNormalization()
        self.a5 = ELU(alpha=1.0)
        # the output shape of this layer is (bz, T, 8, 64)

        self.Reshape_1 = Reshape
        self.conv1d_1 = Conv1D(filters=256, kernel_size=1, strides=1, data_format= 'channels_last', padding= 'same')

        # the output shape of this layer is (bz, T, 256)
        #GLU units
        #block 1
        self.g1_1 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=1,
                               data_format='channels_last', padding='same')
        self.g1_2 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=2,
                             data_format='channels_last', padding='same')
        self.g1_3 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=4,
                             data_format='channels_last', padding='same')
        self.g1_4 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=8,
                             data_format='channels_last', padding='same')
        self.g1_5 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=16,
                             data_format='channels_last', padding='same')

        # block 2
        self.g2_1 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=1,
                             data_format='channels_last', padding='same')
        self.g2_2 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=2,
                             data_format='channels_last', padding='same')
        self.g2_3 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=4,
                             data_format='channels_last', padding='same')
        self.g2_4 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=8,
                             data_format='channels_last', padding='same')
        self.g2_5 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=16,
                             data_format='channels_last', padding='same')

        # block 3
        self.g3_1 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=1,
                             data_format='channels_last', padding='same')
        self.g3_2 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=2,
                             data_format='channels_last', padding='same')
        self.g3_3 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=4,
                             data_format='channels_last', padding='same')
        self.g3_4 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=8,
                             data_format='channels_last', padding='same')
        self.g3_5 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=16,
                             data_format='channels_last', padding='same')

        # block 4
        self.g4_1 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=1,
                             data_format='channels_last', padding='same')
        self.g4_2 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=2,
                             data_format='channels_last', padding='same')
        self.g4_3 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=4,
                             data_format='channels_last', padding='same')
        self.g4_4 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=8,
                             data_format='channels_last', padding='same')
        self.g4_5 = gate_dilation(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1, kernel_size_l2=self.gate_kernel_2, dilation_rate=16,
                             data_format='channels_last', padding='same')
        self.product_g1_1 = tf.concat
        self.product_g1_2 = tf.concat
        self.product_g1_3 = tf.concat
        self.product_g1_4 = tf.concat
        self.m1_conv1d = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last')
        self.product_g2_1 = tf.concat
        self.product_g2_2 = tf.concat
        self.product_g2_3 = tf.concat
        self.product_g2_4 = tf.concat
        self.m2_conv1d = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last')
        self.product_g3_1 = tf.concat
        self.product_g3_2 = tf.concat
        self.product_g3_3 = tf.concat
        self.product_g3_4 = tf.concat
        self.m3_conv1d = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last')
        self.product_g4_1 = tf.concat
        self.product_g4_2 = tf.concat
        self.product_g4_3 = tf.concat
        self.product_g4_4 = tf.concat
        self.m4_conv1d = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last')
        self.product_total12 = tf.concat
        self.product_total123 = tf.concat
        self.product_total1234 = tf.concat
        self.product_total = tf.concat
        self.m5_conv1d = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last')

        ##the output shape is [batch_size, nfft/2, T, 128]

        self.product_c1d1 = tf.concat
        self.product_c2d2 = tf.concat
        self.product_c3d3 = tf.concat
        self.product_c4d4 = tf.concat
        self.product_c5d5 = tf.concat

        self.conv1d_2 = Conv1D(filters=512, kernel_size=1, strides=1, data_format= 'channels_last', padding= 'same')
        self.Reshape_2 = Reshape

        self.d5 = Conv2DTranspose(filters=32, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                                data_format='channels_last')
        self.db5 = BatchNormalization()
        self.da5 = ELU(alpha=1.0)
        self.d4 = Conv2DTranspose(filters=16, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                                data_format='channels_last')
        self.db4 = BatchNormalization()
        self.da4 = ELU(alpha=1.0)
        self.d3 = Conv2DTranspose(filters=8, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                                data_format='channels_last')
        self.db3 = BatchNormalization()
        self.da3 = ELU(alpha=1.0)
        self.d2 = Conv2DTranspose(filters=4, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                                data_format='channels_last')
        self.db2 = BatchNormalization()
        self.da2 = ELU(alpha=1.0)
        self.d1 = Conv2DTranspose(filters=1, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                                data_format='channels_last')
        self.db1 = BatchNormalization()
        self.da1 = tf.keras.layers.LeakyReLU()
        # self.da1 = Activation('linear')
        # self.da1 = Activation('tanh')
        self.Reshape_3 = tf.squeeze
        self.conv1d_out = Conv1D(filters=256, kernel_size=3, strides=1, padding='same')
        self.a6 = Activation('tanh')
    def call(self, inputs):
        input = self.inp_reshape(input= inputs, axis= -1)
        x = self.c1(input)
        x = self.b1(x)
        x_c1 = self.a1(x)
        #(bz, 249, 127, 4)
        x = self.c2(x_c1)
        x = self.b2(x)
        x_c2 = self.a2(x)
        #(bz, 249, 63, 8)
        x = self.c3(x_c2)
        x = self.b3(x)
        x_c3 = self.a3(x)
        #(bz, 249, 31, 16)
        x = self.c4(x_c3)
        x = self.b4(x)
        x_c4 = self.a4(x)
        #(bz, 249, 15, 32)
        x = self.c5(x_c4)
        x = self.b5(x)
        x_c5 = self.a5(x)
        # the output shape of this layer is (bz, 249, 8, 64)

        reshape_xc5 = self.Reshape_1((self.feature_timeaxis, 8 * 64))(x_c5)
        x2glu = self.conv1d_1(reshape_xc5)

        # the output shape of this layer is (bz, T, 256)

        x_res1_1, x_jump1_1 = self.g1_1(x2glu)
        x_res1_2, x_jump1_2 = self.g1_2(x_res1_1)
        x_res1_3, x_jump1_3 = self.g1_3(x_res1_2)
        x_res1_4, x_jump1_4 = self.g1_4(x_res1_3)
        x_res1_5, x_jump1_5 = self.g1_5(x_res1_4)

        total_1 = self.product_g1_1([x_jump1_1, x_jump1_2], axis=-1)
        total_1 = self.product_g1_2([total_1, x_jump1_3], axis=-1)
        total_1 = self.product_g1_3([total_1, x_jump1_4], axis=-1)
        total_1 = self.product_g1_3([total_1, x_jump1_5], axis=-1)
        total_1 = self.m1_conv1d(total_1)

        x_res2_1, x_jump2_1 = self.g2_1(x_res1_5)
        x_res2_2, x_jump2_2 = self.g2_2(x_res2_1)
        x_res2_3, x_jump2_3 = self.g2_3(x_res2_2)
        x_res2_4, x_jump2_4 = self.g2_4(x_res2_3)
        x_res2_5, x_jump2_5 = self.g2_5(x_res2_4)

        total_2 = self.product_g2_1([x_jump2_1, x_jump2_2], axis=-1)
        total_2 = self.product_g2_2([total_2, x_jump2_3], axis=-1)
        total_2 = self.product_g2_3([total_2, x_jump2_4], axis=-1)
        total_2 = self.product_g2_3([total_2, x_jump2_5], axis=-1)
        total_2 = self.m2_conv1d(total_2)

        x_res3_1, x_jump3_1 = self.g3_1(x_res2_5)
        x_res3_2, x_jump3_2 = self.g3_2(x_res3_1)
        x_res3_3, x_jump3_3 = self.g3_3(x_res3_2)
        x_res3_4, x_jump3_4 = self.g3_4(x_res3_3)
        x_res3_5, x_jump3_5 = self.g3_5(x_res3_4)

        total_3 = self.product_g3_1([x_jump3_1, x_jump3_2], axis=-1)
        total_3 = self.product_g3_2([total_3, x_jump3_3], axis=-1)
        total_3 = self.product_g3_3([total_3, x_jump3_4], axis=-1)
        total_3 = self.product_g3_3([total_3, x_jump3_5], axis=-1)
        total_3 = self.m3_conv1d(total_3)

        x_res4_1, x_jump4_1 = self.g4_1(x_res3_5)
        x_res4_2, x_jump4_2 = self.g4_2(x_res4_1)
        x_res4_3, x_jump4_3 = self.g4_3(x_res4_2)
        x_res4_4, x_jump4_4 = self.g4_4(x_res4_3)
        x_res4_5, x_jump4_5 = self.g4_5(x_res4_4)

        total_4 = self.product_g4_1([x_jump4_1, x_jump4_2], axis=-1)
        total_4 = self.product_g4_2([total_4, x_jump4_3], axis=-1)
        total_4 = self.product_g4_3([total_4, x_jump4_4], axis=-1)
        total_4 = self.product_g4_3([total_4, x_jump4_5], axis=-1)
        total_4 = self.m4_conv1d(total_4)

        total_12 = self.product_total12([total_1, total_2], axis=-1)
        total_123 = self.product_total123([total_12, total_3], axis=-1)
        total_1234 = self.product_total123([total_123, total_4], axis=-1)
        total = self.product_total([total_1234, x_res4_5], axis=-1)
        # total = self.product_total([total_1234, x_res1_5, x_res2_5, x_res3_5, x_res4_5], axis=-1)
        # total = self.m5_conv1d(total)

        x = self.conv1d_2(total)

        x = self.Reshape_2((self.feature_timeaxis, 8, 64))(x)
        # the shape of x is (bz, T, 8, 64)

        x = self.product_c5d5([x, x_c5], axis=-1)
        x = self.d5(x)
        x = self.db5(x)
        x_d5 = self.da5(x)

        x = self.product_c4d4([x_d5, x_c4], axis=-1)
        x = self.d4(x)
        x = self.db4(x)
        x_d4 = self.da4(x)

        x = self.product_c3d3([x_d4, x_c3], axis=-1)
        x = self.d3(x)
        x = self.db3(x)
        x_d3 = self.da3(x)

        x = self.product_c2d2([x_d3, x_c2], axis=-1)
        x = self.d2(x)
        x = self.db2(x)
        x_d2 = self.da2(x)

        x = self.product_c1d1([x_d2, x_c1], axis=-1)
        x = self.d1(x)
        x = self.db1(x)
        x_d1 = self.da1(x)

        x = self.Reshape_3(x_d1, axis= -1)

        # output = self.conv1d_out(x)
        # output = self.a6(output)
        return x


model_unit = Project_1129(gate_filter=256, kernel_size=32, strides=(1, 2), gate_kernel_1= 5, gate_kernel_2=1, feature_timeaxis=124)





