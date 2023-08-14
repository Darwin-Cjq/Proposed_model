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
from keras.layers import Dropout, Conv2D, Conv1D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D, Conv2DTranspose, Lambda, Reshape, ELU
import Complexnn as complexnn
import tensorflow.keras.backend as K
import complex_transconv
import os
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.keras
# import evaluation

class gate_dilation(tf.keras.Model):
    def __init__(self, filter, kernel_size_l1, kernel_size_l2, dilation_rate, padding='same'):
        super(gate_dilation, self).__init__()
        self.filter = filter
        self.kernel_size_l1 = kernel_size_l1
        self.kernel_size_l2 = kernel_size_l2
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.a1_1 = Activation('linear')
        self.a1_2 = Activation('sigmoid')
        self.d1_1 = complexnn.conv.ComplexConv1D(filters=self.filter, kernel_size=self.kernel_size_l1, strides=1,
                           dilation_rate=self.dilation_rate, padding= self.padding)
        self.d1_2 = complexnn.conv.ComplexConv1D(filters=self.filter, kernel_size=self.kernel_size_l1, strides=1,
                           dilation_rate=self.dilation_rate, padding= self.padding)
        self.product_1 = tf.multiply

    def call(self, inputs):
        x1_1 = self.d1_1(inputs)
        x1_2 = self.d1_2(inputs)
        ax1_1 = self.a1_1(x1_1)
        ax1_2 = self.a1_2(x1_2)
        outcome = self.product_1(ax1_2, ax1_1)

        return outcome

class gate_middle(tf.keras.Model):
    def __init__(self, filter, kernel_size_l1, kernel_size_l2, dilation_rate, padding):
        super(gate_middle, self).__init__()
        self.filter = filter
        self.kernel_size_l1 = kernel_size_l1
        self.kernel_size_l2 = kernel_size_l2
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.a1_1 = ELU(alpha=1.0)
        self.a1_2 = Activation('sigmoid')

        self.d1_1 = complexnn.conv.ComplexConv1D(filters=self.filter, kernel_size=self.kernel_size_l1, strides=1,
                           dilation_rate=self.dilation_rate, padding= self.padding)
        self.d1_2 = complexnn.conv.ComplexConv1D(filters=self.filter, kernel_size=self.kernel_size_l1, strides=1,
                           dilation_rate=self.dilation_rate, padding= self.padding)

        self.d2_1 = complexnn.conv.ComplexConv1D(filters=self.filter, kernel_size=self.kernel_size_l2, strides=1,
                           dilation_rate=self.dilation_rate, padding= self.padding)
        self.d2_2 = complexnn.conv.ComplexConv1D(filters=self.filter, kernel_size=self.kernel_size_l2, strides=1,
                           dilation_rate=self.dilation_rate, padding= self.padding)
        self.b2_1 = complexnn.ComplexLayerNorm()
        self.b2_2 = complexnn.ComplexLayerNorm()
        self.a2_1 = tf.keras.layers.LeakyReLU()
        self.a2_2 = tf.keras.layers.LeakyReLU()

        # self.product_3 = tf.concat
    def call(self, inputs):
        x1_1 = self.d1_1(inputs)
        x1_2 = self.d1_2(inputs)
        ax1_1 = self.a1_1(x1_1)
        ax1_2 = self.a1_2(x1_2)
        outcome_l1 = tf.multiply(ax1_2, ax1_1)
        x2_1 = self.d2_1(outcome_l1)
        x2_2 = self.d2_2(outcome_l1)
        b2_1 = self.b2_1(x2_1)
        b2_2 = self.b2_2(x2_2)
        jump_out = self.a2_1(b2_2)
        resi_out = tf.add(inputs, b2_1)
        resi_out = self.a2_2(resi_out)
        # resi_jump = self.product_3([resi_out, jump_out], axis=-1)
        return resi_out, jump_out



class tradition_glu(tf.keras.Model):
    def __init__(self, kernel_size, strides, filters, gate_filter, gate_kernel_1, gate_kernel_2):
        super(tradition_glu, self).__init__()
        self.gate_filter = gate_filter
        self.gate_kernel_1 = gate_kernel_1
        self.gate_kernel_2 = gate_kernel_2
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        # self.glu_kernel_1 = glu_kernel_1
        # self.glu_kernel_2 = glu_kernel_2

        # Encoder layers
        # Encoder layers
        self.conv1d_0 = complexnn.conv.ComplexConv1D(filters=self.filters[0], kernel_size=self.kernel_size[0],
                                                     strides=self.strides[0], padding='same')
        self.LN_0 = complexnn.ComplexLayerNorm()
        self.act_0 = ELU(alpha=1.0)
        self.en_block0 = tf.keras.models.Sequential([self.conv1d_0, self.LN_0, self.act_0])
        ###
        self.conv1d_1 = complexnn.conv.ComplexConv1D(filters=self.filters[1], kernel_size=self.kernel_size[1],
                                                     strides=self.strides[1],
                                                     padding='same')
        self.LN_1 = complexnn.ComplexLayerNorm()
        self.act_1 = ELU(alpha=1.0)
        self.en_block1 = tf.keras.models.Sequential([self.conv1d_1, self.LN_1, self.act_1])
        ###
        self.conv1d_2 = complexnn.conv.ComplexConv1D(filters=self.filters[2], kernel_size=self.kernel_size[2],
                                                     strides=self.strides[2],
                                                     padding='same')
        self.LN_2 = complexnn.ComplexLayerNorm()
        self.act_2 = ELU(alpha=1.0)
        self.en_block2 = tf.keras.models.Sequential([self.conv1d_2, self.LN_2, self.act_2])
        ###
        self.conv1d_3 = complexnn.conv.ComplexConv1D(filters=self.filters[3], kernel_size=self.kernel_size[3],
                                                        strides=self.strides[3], padding='same')
        self.LN_3 = complexnn.ComplexLayerNorm()
        self.act_3 = ELU(alpha=1.0)
        self.en_block3 = tf.keras.models.Sequential([self.conv1d_3, self.LN_3, self.act_3])
        ###
        self.conv1d_4 = complexnn.conv.ComplexConv1D(filters=self.filters[4], kernel_size=self.kernel_size[4],
                                                        strides=self.strides[4], padding='same')
        self.LN_4 = complexnn.ComplexLayerNorm()
        self.act_4 = ELU(alpha=1.0)
        self.en_block4 = tf.keras.models.Sequential([self.conv1d_4, self.LN_4, self.act_4])

        # GLU units
        # block 1
        self.g1_1 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=1, padding='same')
        self.g1_2 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=2, padding='same')
        self.g1_3 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=4, padding='same')
        self.g1_4 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=8, padding='same')
        self.g1_5 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=16, padding='same')

        # block 2
        self.g2_1 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=1, padding='same')
        self.g2_2 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=2, padding='same')
        self.g2_3 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=4, padding='same')
        self.g2_4 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=8, padding='same')
        self.g2_5 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=16, padding='same')

        # block 3
        self.g3_1 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=1, padding='same')
        self.g3_2 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=2, padding='same')
        self.g3_3 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=4, padding='same')
        self.g3_4 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=8, padding='same')
        self.g3_5 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=16, padding='same')

        # block 4
        self.g4_1 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=1, padding='same')
        self.g4_2 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=2, padding='same')
        self.g4_3 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=4, padding='same')
        self.g4_4 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=8, padding='same')
        self.g4_5 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=16, padding='same')

        self.m1_conv1d = complexnn.conv.ComplexConv1D(filters=64, kernel_size=1, strides=1, padding='same')

        self.m2_conv1d = complexnn.conv.ComplexConv1D(filters=64, kernel_size=1, strides=1, padding='same')

        self.m3_conv1d = complexnn.conv.ComplexConv1D(filters=64, kernel_size=1, strides=1, padding='same')

        self.m4_conv1d = complexnn.conv.ComplexConv1D(filters=64, kernel_size=1, strides=1, padding='same')

        self.m5_conv1d = complexnn.conv.ComplexConv1D(filters=64, kernel_size=1, strides=1, padding='same')
        self.conv1d_2 = complexnn.conv.ComplexConv1D(filters=8, kernel_size=1, strides=1, padding='same')
        ##the output shape is [batch_size, nfft/2, T, 128]


        # Decoder layers
        self.deconv1d_4 = complex_transconv.ComplexTransposedConv1D(filters=self.filters[3],
                                                                    kernel_size=self.kernel_size[4],
                                                                    strides=self.strides[4],
                                                                    padding='same')
        self.deLN_4 = complexnn.ComplexLayerNorm()
        self.deact_4 = ELU(alpha=1.0)
        self.de_block4 = tf.keras.models.Sequential([self.deconv1d_4, self.deLN_4, self.deact_4])

        ###
        self.deconv1d_3 = complex_transconv.ComplexTransposedConv1D(filters=self.filters[2],
                                                                    kernel_size=self.kernel_size[3],
                                                                    strides=self.strides[3],
                                                                    padding='same',
                                                                    output_padding=1)
        self.deLN_3 = complexnn.ComplexLayerNorm()
        self.deact_3 = ELU(alpha=1.0)
        self.de_block3 = tf.keras.models.Sequential([self.deconv1d_3, self.deLN_3, self.deact_3])

        ###
        self.deconv1d_2 = complex_transconv.ComplexTransposedConv1D(filters=self.filters[1],
                                                                    kernel_size=self.kernel_size[2],
                                                                    strides=self.strides[2],
                                                                    padding='same')
        self.deLN_2 = complexnn.ComplexLayerNorm()
        self.deact_2 = ELU(alpha=1.0)
        self.de_block2 = tf.keras.models.Sequential([self.deconv1d_2, self.deLN_2, self.deact_2])

        ###
        self.deconv1d_1 = complex_transconv.ComplexTransposedConv1D(filters=self.filters[0],
                                                                    kernel_size=self.kernel_size[1],
                                                                    strides=self.strides[1],
                                                                    padding='same')
        self.deLN_1 = complexnn.ComplexLayerNorm()
        self.deact_1 = ELU(alpha=1.0)
        self.de_block1 = tf.keras.models.Sequential([self.deconv1d_1, self.deLN_1, self.deact_1])

        ###
        self.deconv1d_0 = complex_transconv.ComplexTransposedConv1D(filters=self.filters[-1],
                                                                    kernel_size=self.kernel_size[0],
                                                                    strides=self.strides[0],
                                                                    padding='same')
        # self.deact_0 = ELU(alpha=1.0)
        self.de_block0 = tf.keras.models.Sequential([self.deconv1d_0])

    def call(self, inputs, training=None, mask=None):
        # x = tf.expand_dims(inputs, axis=-1)
        x_0 = self.en_block0(inputs)
        x_1 = self.en_block1(x_0)
        x_2 = self.en_block2(x_1)
        x_3 = self.en_block3(x_2)
        x_4 = self.en_block4(x_3)
        # x_4 shape is [batch_size, 2, 16]

        x_res1_1, x_jump1_1 = self.g1_1(x_4)
        x_res1_2, x_jump1_2 = self.g1_2(x_res1_1)
        x_res1_3, x_jump1_3 = self.g1_3(x_res1_2)
        x_res1_4, x_jump1_4 = self.g1_4(x_res1_3)
        x_res1_5, x_jump1_5 = self.g1_5(x_res1_4)

        total_1 = tf.concat([x_jump1_1, x_jump1_2], axis=-1)
        total_1 = tf.concat([total_1, x_jump1_3], axis=-1)
        total_1 = tf.concat([total_1, x_jump1_4], axis=-1)
        total_1 = tf.concat([total_1, x_jump1_5], axis=-1)
        total_1 = self.m1_conv1d(total_1)

        x_res2_1, x_jump2_1 = self.g2_1(x_res1_5)
        x_res2_2, x_jump2_2 = self.g2_2(x_res2_1)
        x_res2_3, x_jump2_3 = self.g2_3(x_res2_2)
        x_res2_4, x_jump2_4 = self.g2_4(x_res2_3)
        x_res2_5, x_jump2_5 = self.g2_5(x_res2_4)

        total_2 = tf.concat([x_jump2_1, x_jump2_2], axis=-1)
        total_2 = tf.concat([total_2, x_jump2_3], axis=-1)
        total_2 = tf.concat([total_2, x_jump2_4], axis=-1)
        total_2 = tf.concat([total_2, x_jump2_5], axis=-1)
        total_2 = self.m2_conv1d(total_2)

        x_res3_1, x_jump3_1 = self.g3_1(x_res2_5)
        x_res3_2, x_jump3_2 = self.g3_2(x_res3_1)
        x_res3_3, x_jump3_3 = self.g3_3(x_res3_2)
        x_res3_4, x_jump3_4 = self.g3_4(x_res3_3)
        x_res3_5, x_jump3_5 = self.g3_5(x_res3_4)

        total_3 = tf.concat([x_jump3_1, x_jump3_2], axis=-1)
        total_3 = tf.concat([total_3, x_jump3_3], axis=-1)
        total_3 = tf.concat([total_3, x_jump3_4], axis=-1)
        total_3 = tf.concat([total_3, x_jump3_5], axis=-1)
        total_3 = self.m3_conv1d(total_3)

        # x_res4_1, x_jump4_1 = self.g4_1(x_res3_5)
        # x_res4_2, x_jump4_2 = self.g4_2(x_res4_1)
        # x_res4_3, x_jump4_3 = self.g4_3(x_res4_2)
        # x_res4_4, x_jump4_4 = self.g4_4(x_res4_3)
        # x_res4_5, x_jump4_5 = self.g4_5(x_res4_4)

        # total_4 = tf.concat([x_jump4_1, x_jump4_2], axis=-1)
        # total_4 = tf.concat([total_4, x_jump4_3], axis=-1)
        # total_4 = tf.concat([total_4, x_jump4_4], axis=-1)
        # total_4 = tf.concat([total_4, x_jump4_5], axis=-1)
        # total_4 = self.m4_conv2d(total_4)

        total_12 = tf.concat([total_1, total_2], axis=-1)
        total_123 = tf.concat([total_12, total_3], axis=-1)
        # total_1234 = tf.concat([total_123, total_4], axis=-1)
        x = self.conv1d_2(total_123)
        total = x_res3_5 + x
        # total = self.product_total([total_1234, x_res1_5, x_res2_5, x_res3_5, x_res4_5], axis=-1)
        # total = self.m5_conv1d(total)
        de_feed4 = tf.concat([x_4, total], axis=-1)
        y_4 = self.de_block4(de_feed4)
        de_feed3 = tf.concat([x_3, y_4], axis=-1)
        y_3 = self.de_block3(de_feed3)
        de_feed2 = tf.concat([x_2, y_3], axis=-1)
        y_2 = self.de_block2(de_feed2)
        de_feed1 = tf.concat([x_1, y_2], axis=-1)
        y_1 = self.de_block1(de_feed1)
        de_feed0 = tf.concat([x_0, y_1], axis=-1)
        y_0 = self.de_block0(de_feed0)
        return y_0


model = tradition_glu(kernel_size=[3, 4, 4, 4, 4], strides=[1, 2, 2, 2, 2], filters=[128, 64, 32, 16, 8, 256],
                gate_filter=8, gate_kernel_1=5, gate_kernel_2=1)
# input_darwin = K.random_uniform(shape=[2, 124, 512], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res = model(input_darwin)
# print(res.shape)
# exit()



