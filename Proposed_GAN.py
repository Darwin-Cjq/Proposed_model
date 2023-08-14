import tensorflow as tf
from keras import Model
from tensorflow_addons import layers as addon_layers
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, Conv2D, Conv1D, BatchNormalization, Activation, Conv2DTranspose, Lambda, Reshape, ELU
import keras.backend as K
from tensorflow_addons.optimizers import AdamW
import os
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import keras
import evaluation
import librosa
from scipy.io.wavfile import write


#load the data

# X_train, X_test, y_train, y_test = train_test_split(obj_noisy, obj_pure, test_size=0.2, random_state=42, shuffle=False)

BATCH_SIZE = 32
Epochs = 10


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
        self.a1_1 = Activation('linear')
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


class Generator(Model):
    def __init__(self, block_num, gate_filter,kernel_size, strides):
        super(Generator, self).__init__()
        self.gate_filter = gate_filter
        self.block_num = block_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.inp_reshape = tf.expand_dims

        self.c1 = Conv2D(filters=4, kernel_size= self.kernel_size, strides= self.strides, padding= 'same',
                                     data_format= 'channels_last')
        self.b1 = BatchNormalization()
        self.a1 = ELU(alpha=1.0)

        self.c2 = Conv2D(filters=8, kernel_size= self.kernel_size, strides= self.strides, padding='same',
                                     data_format='channels_last')
        self.b2 = BatchNormalization()
        self.a2 = ELU(alpha=1.0)

        self.c3 = Conv2D(filters=16, kernel_size= self.kernel_size, strides= self.strides, padding='same',
                                     data_format='channels_last')
        self.b3 = BatchNormalization()
        self.a3 = ELU(alpha=1.0)

        self.c4 = Conv2D(filters=32, kernel_size= self.kernel_size, strides= self.strides, padding='same',
                                     data_format='channels_last')
        self.b4 = BatchNormalization()
        self.a4 = ELU(alpha=1.0)

        self.c5 = Conv2D(filters=64, kernel_size= self.kernel_size, strides= self.strides, padding= 'same',
                                     data_format= 'channels_last')
        self.b5 = BatchNormalization()
        self.a5 = ELU(alpha=1.0)
        # the output shape of this layer is (bz, T, 8, 64)

        self.Reshape_1 = Reshape
        self.conv1d_1 = Conv1D(filters=256, kernel_size=1, strides=1, data_format= 'channels_last', padding= 'same')
        # the output shape of this layer is (bz, T, 256)
        #GLU units
        #block 1
        self.g1_1 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=1,
                               data_format='channels_last', padding='same')
        self.g1_2 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=2,
                             data_format='channels_last', padding='same')
        self.g1_3 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=4,
                             data_format='channels_last', padding='same')
        self.g1_4 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=8,
                             data_format='channels_last', padding='same')
        self.g1_5 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=16,
                             data_format='channels_last', padding='same')

        # block 2
        self.g2_1 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=1,
                             data_format='channels_last', padding='same')
        self.g2_2 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=2,
                             data_format='channels_last', padding='same')
        self.g2_3 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=4,
                             data_format='channels_last', padding='same')
        self.g2_4 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=8,
                             data_format='channels_last', padding='same')
        self.g2_5 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=16,
                             data_format='channels_last', padding='same')

        # block 3
        self.g3_1 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=1,
                             data_format='channels_last', padding='same')
        self.g3_2 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=2,
                             data_format='channels_last', padding='same')
        self.g3_3 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=4,
                             data_format='channels_last', padding='same')
        self.g3_4 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=8,
                             data_format='channels_last', padding='same')
        self.g3_5 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=16,
                             data_format='channels_last', padding='same')

        # block 4
        self.g4_1 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=1,
                             data_format='channels_last', padding='same')
        self.g4_2 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=2,
                             data_format='channels_last', padding='same')
        self.g4_3 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=4,
                             data_format='channels_last', padding='same')
        self.g4_4 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=8,
                             data_format='channels_last', padding='same')
        self.g4_5 = gate_dilation(filter=self.gate_filter, kernel_size_l1=5, kernel_size_l2=1, dilation_rate=16,
                             data_format='channels_last', padding='same')
        self.product_g1_1 = tf.add
        self.product_g1_2 = tf.add
        self.product_g1_3 = tf.add
        self.product_g1_4 = tf.add
        self.product_g2_1 = tf.add
        self.product_g2_2 = tf.add
        self.product_g2_3 = tf.add
        self.product_g2_4 = tf.add
        self.product_g3_1 = tf.add
        self.product_g3_2 = tf.add
        self.product_g3_3 = tf.add
        self.product_g3_4 = tf.add
        self.product_g4_1 = tf.add
        self.product_g4_2 = tf.add
        self.product_g4_3 = tf.add
        self.product_g4_4 = tf.add
        self.product_total12 = tf.add
        self.product_total123 = tf.add
        self.product_total1234 = tf.add
        self.product_total = tf.add

        ##the output shape is [batch_size, nfft/2, T, 128]

        self.product_c1d1 = tf.concat
        self.product_c2d2 = tf.concat
        self.product_c3d3 = tf.concat
        self.product_c4d4 = tf.concat
        self.product_c5d5 = tf.concat

        self.conv1d_2 = Conv1D(filters=512, kernel_size=1, strides=1, data_format= 'channels_last', padding= 'same')
        self.Reshape_2 = Reshape

        self.d5 = Conv2DTranspose(filters=32, kernel_size= self.kernel_size, strides= self.strides, padding='same',
                                                data_format='channels_last')
        self.db5 = BatchNormalization()
        self.da5 = ELU(alpha=1.0)
        self.d4 = Conv2DTranspose(filters=16, kernel_size= self.kernel_size, strides= self.strides, padding='same',
                                                data_format='channels_last')
        self.db4 = BatchNormalization()
        self.da4 = ELU(alpha=1.0)
        self.d3 = Conv2DTranspose(filters=8, kernel_size= self.kernel_size, strides= self.strides, padding='same',
                                                data_format='channels_last')
        self.db3 = BatchNormalization()
        self.da3 = ELU(alpha=1.0)
        self.d2 = Conv2DTranspose(filters=4, kernel_size= self.kernel_size, strides= self.strides, padding='same',
                                                data_format='channels_last')
        self.db2 = BatchNormalization()
        self.da2 = ELU(alpha=1.0)
        self.d1 = Conv2DTranspose(filters=1, kernel_size= self.kernel_size, strides= self.strides, padding='same',
                                                data_format='channels_last')
        self.db1 = BatchNormalization()
        # self.da1 = tf.keras.layers.LeakyReLU()
        self.da1 = Activation('linear')
        self.Reshape_3 = tf.squeeze

    def call(self, inputs):
        inputs = self.inp_reshape(input= inputs, axis= -1)
        x = self.c1(inputs)
        x = self.b1(x)
        x_c1 = self.a1(x)
        x = self.c2(x_c1)
        x = self.b2(x)
        x_c2 = self.a2(x)
        x = self.c3(x_c2)
        x = self.b3(x)
        x_c3 = self.a3(x)
        x = self.c4(x_c3)
        x = self.b4(x)
        x_c4 = self.a4(x)
        x = self.c5(x_c4)
        x = self.b5(x)
        x_c5 = self.a5(x)
        # the output shape of this layer is (bz, T, 8, 64)

        reshape_xc5 = self.Reshape_1((249, 8 * 64))(x_c5)
        x2glu = self.conv1d_1(reshape_xc5)
        # the output shape of this layer is (bz, T, 256)

        x_res1_1, x_jump1_1 = self.g1_1(x2glu)
        x_res1_2, x_jump1_2 = self.g1_2(x_res1_1)
        x_res1_3, x_jump1_3 = self.g1_3(x_res1_2)
        x_res1_4, x_jump1_4 = self.g1_4(x_res1_3)
        x_res1_5, x_jump1_5 = self.g1_5(x_res1_4)

        total_1 = self.product_g1_1(x_jump1_1, x_jump1_2)
        total_1 = self.product_g1_2(total_1, x_jump1_3)
        total_1 = self.product_g1_3(total_1, x_jump1_4)
        total_1 = self.product_g1_3(total_1, x_jump1_5)

        x_res2_1, x_jump2_1 = self.g2_1(x_res1_5)
        x_res2_2, x_jump2_2 = self.g2_2(x_res2_1)
        x_res2_3, x_jump2_3 = self.g2_3(x_res2_2)
        x_res2_4, x_jump2_4 = self.g2_4(x_res2_3)
        x_res2_5, x_jump2_5 = self.g2_5(x_res2_4)

        total_2 = self.product_g2_1(x_jump2_1, x_jump2_2)
        total_2 = self.product_g2_2(total_2, x_jump2_3)
        total_2 = self.product_g2_3(total_2, x_jump2_4)
        total_2 = self.product_g2_3(total_2, x_jump2_5)

        x_res3_1, x_jump3_1 = self.g3_1(x_res2_5)
        x_res3_2, x_jump3_2 = self.g3_2(x_res3_1)
        x_res3_3, x_jump3_3 = self.g3_3(x_res3_2)
        x_res3_4, x_jump3_4 = self.g3_4(x_res3_3)
        x_res3_5, x_jump3_5 = self.g3_5(x_res3_4)

        total_3 = self.product_g3_1(x_jump3_1, x_jump3_2)
        total_3 = self.product_g3_2(total_3, x_jump3_3)
        total_3 = self.product_g3_3(total_3, x_jump3_4)
        total_3 = self.product_g3_3(total_3, x_jump3_5)

        x_res4_1, x_jump4_1 = self.g4_1(x_res3_5)
        x_res4_2, x_jump4_2 = self.g4_2(x_res4_1)
        x_res4_3, x_jump4_3 = self.g4_3(x_res4_2)
        x_res4_4, x_jump4_4 = self.g4_4(x_res4_3)
        x_res4_5, x_jump4_5 = self.g4_5(x_res4_4)

        total_4 = self.product_g4_1(x_jump4_1, x_jump4_2)
        total_4 = self.product_g4_2(total_4, x_jump4_3)
        total_4 = self.product_g4_3(total_4, x_jump4_4)
        total_4 = self.product_g4_3(total_4, x_jump4_5)

        total_12 = self.product_total12(total_1, total_2)
        total_123 = self.product_total123(total_12, total_3)
        total_1234 = self.product_total123(total_123, total_4)
        total = self.product_total(total_1234, x_res4_5)
        # the shape of total is (bz, 126, 256)

        # x = self.product_c5d5(total, x_c5)
        x = self.conv1d_2(total)
        # the shape of x is (bz, 126, 512)
        x = self.Reshape_2((249, 8, 64))(x)
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
        output = x

        return output

generator = Generator(block_num=4, gate_filter=256, kernel_size= (32, 32), strides= (1, 2))
# input_darwin = keras.backend.random_uniform(shape= [2, 249, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res = generator(input_darwin)
# print(res.shape)#output shape is (bz, 126, 256)
# exit()

def discriminator_block(input):
    conv1 = addon_layers.WeightNormalization(
        layers.Conv1D(16, 8, 1, "same"), data_init=False
    )(input) #output shape is (bz, 126, 16)
    BN1 = BatchNormalization()(conv1)
    lrelu1 = layers.LeakyReLU()(BN1)

    conv2 = addon_layers.WeightNormalization(
        layers.Conv1D(64, 8, 2, "same", groups=4), data_init=False
    )(lrelu1)#output shape is (bz, 126/2, 64)
    BN2 = BatchNormalization()(conv2)
    lrelu2 = layers.LeakyReLU()(BN2)

    conv3 = addon_layers.WeightNormalization(
        layers.Conv1D(256, 8, 2, "same", groups=16), data_init=False
    )(lrelu2)#output shape is (bz, 126/4, 256)
    BN3 = BatchNormalization()(conv3)
    lrelu3 = layers.LeakyReLU()(BN3)

    conv4 = addon_layers.WeightNormalization(
        layers.Conv1D(1024, 8, 2, "same", groups=64), data_init=False
    )(lrelu3)#output shape is (bz, 126/8, 1024)
    BN4 = BatchNormalization()(conv4)
    lrelu4 = layers.LeakyReLU()(BN4)

    conv5 = addon_layers.WeightNormalization(
        layers.Conv1D(1024, 8, 2, "same", groups=256), data_init=False
    )(lrelu4)#output shape is (bz, 126/16, 1024)
    BN5 = BatchNormalization()(conv5)
    lrelu5 = layers.LeakyReLU()(BN5)

    conv6 = addon_layers.WeightNormalization(
        layers.Conv1D(1024, 4, 1, "same"), data_init=False
    )(lrelu5)#output shape is (bz, 126/16, 1024)
    BN6 = BatchNormalization()(conv6)
    lrelu6 = layers.LeakyReLU()(BN6)

    conv7 = addon_layers.WeightNormalization(
        layers.Conv1D(1, 2, 1, "same"), data_init=False
    )(lrelu6)#output shape is (bz, 126/16, 1)

    return [lrelu1, lrelu2, lrelu3, lrelu4, lrelu5, lrelu6, conv7]

"""
### Create the discriminator
"""


def create_discriminator(input_shape):
    inp = keras.Input(input_shape)
    out_map1 = discriminator_block(inp)
    pool1 = layers.AveragePooling1D()(inp)
    out_map2 = discriminator_block(pool1)
    # pool2 = layers.AveragePooling1D()(pool1)
    # out_map3 = discriminator_block(pool2)
    flat = tf.keras.layers.Flatten()(out_map2[-1])
    dense = tf.keras.layers.Dense(1, use_bias=False)(flat)
    out = Activation('sigmoid')(dense)
    return keras.Model(inp, [inp, out_map1, out_map2, [out]])


input_darwin = keras.backend.random_uniform(shape= [2, 249, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
discriminator = create_discriminator((249, 256))
# res = discriminator(input_darwin)
# print(res.shape)
# exit()


def mean_pred(y_true, y_pred):
    return K.abs(K.mean(y_true - y_pred))

def loss_func(y_pred, y_true):
    loss_1 = evaluation.si_sdr3(y_pred, y_true)
    # loss_2 = K.mean(K.abs(y_pred - y_true), axis=-1)
    return loss_1

def generator_loss(real_pred, fake_pred):
    #the output of disc is [inp, out_map1, out_map2, out_map3, [out]]
    #the output of out_map is [lrelu1, lrelu2, lrelu3, lrelu4, lrelu5, lrelu6, conv7]
    """Loss function for the generator.
    Args:
        real_pred: Tensor, output of the ground truth wave passed through the discriminator.
        fake_pred: Tensor, output of the generator prediction passed through the discriminator.
    Returns:
        Loss for the generator.
    """

    loss = loss_func(fake_pred[0], real_pred[0])

    return loss

mse = keras.losses.MeanSquaredError()
mae = keras.losses.MeanAbsoluteError()

def discriminator_loss(real_pred, fake_pred):
    # the output of disc is [inp, out_map1, out_map2, out_map3, [out]]
    # the output of out_map is [lrelu1, lrelu2, lrelu3, lrelu4, lrelu5, lrelu6, conv7]
    """Implements the discriminator loss.
    Args:
        real_pred: Tensor, output of the ground truth wave passed through the discriminator.
        fake_pred: Tensor, output of the generator prediction passed through the discriminator.
    Returns:
        Discriminator Loss.
    """
    real_loss, fake_loss = [], []
    for i in range(1, len(real_pred)):
        real_loss.append(mse(tf.ones_like(real_pred[i][-1]), real_pred[i][-1]))
        fake_loss.append(mse(tf.zeros_like(fake_pred[i][-1]), fake_pred[i][-1]))

    # Calculating the final discriminator loss after scaling
    disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)
    return disc_loss

class MelGAN(keras.Model):
    def __init__(self, generator, discriminator, **kwargs):

        """MelGAN trainer class
        Args:
            generator: keras.Model, Generator model
            discriminator: keras.Model, Discriminator model
        """
        super().__init__(**kwargs)

        self.generator = generator
        self.discriminator = discriminator

    def compile(
        self,
        gen_optimizer,
        disc_optimizer,
        generator_loss,
        discriminator_loss,
    ):
        """MelGAN compile method.
        Args:
            gen_optimizer: keras.optimizer, optimizer to be used for training
            disc_optimizer: keras.optimizer, optimizer to be used for training
            generator_loss: callable, loss function for generator
            feature_matching_loss: callable, loss function for feature matching
            discriminator_loss: callable, loss function for discriminator
        """
        super().compile()

        # Optimizers
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

        # Losses

        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

        # Trackers
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")

    def train_step(self, batch):
        x_batch_train, y_batch_train = batch
        # x_batch_train.shape = (bz, 128, 251), y_batch_train.shape = (bz, 64000)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generating the audio wave
            gen_audio_wave = generator(x_batch_train, training=True)

            # Generating the features using the discriminator
            fake_pred = discriminator(y_batch_train)
            real_pred = discriminator(gen_audio_wave)

            # Calculating the generator losses
            gen_loss = generator_loss(real_pred, fake_pred)

            # Calculating final generator loss
            gen_fm_loss = gen_loss

            # Calculating the discriminator losses
            disc_loss = discriminator_loss(real_pred, fake_pred)

        # Calculating and applying the gradients for generator and discriminator
        grads_gen = gen_tape.gradient(gen_fm_loss, generator.trainable_weights)
        grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_weights)
        gen_optimizer.apply_gradients(zip(grads_gen, generator.trainable_weights))
        disc_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

        self.gen_loss_tracker.update_state(gen_fm_loss)
        self.disc_loss_tracker.update_state(disc_loss)

        return {
            "gen_loss": self.gen_loss_tracker.result(),
            "disc_loss": self.disc_loss_tracker.result(),
        }

        # implement the call method

    def call(self, inputs, *args, **kwargs):
        x = self.generator(inputs)
        y_pred = self.discriminator(x)
        return y_pred

lr_batch = 2000
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
	[lr_batch * 10, lr_batch * 20], [1e-3, 1e-4, 1e-5])

wd_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
	[lr_batch * 10, lr_batch * 20], [1e-4, 1e-5, 1e-6])
gen_optimizer = AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule)
disc_optimizer = AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule)

# Start training

# generator = Generator(block_num=4, gate_filter=256, kernel_size= (32, 32), strides= (1, 2))
# discriminator = create_discriminator((249, 256))

mel_gan = MelGAN(generator, discriminator)

mel_gan.compile(
    gen_optimizer,
    disc_optimizer,
    generator_loss,
    discriminator_loss,
)

modelweights_save_path = "C:/Users/Darwin.Cui/OneDrive - University of Southampton/Desktop/SE_dilation/MelGAN/parameters/GAN_mag_pink.h5"
if os.path.exists(modelweights_save_path):
    print('-------------load the model weights-----------------')
    # mel_gan.build(input_shape=(None, 249, 256))
    mel_gan.load_weights(modelweights_save_path, by_name= True)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=modelweights_save_path,
                                                 save_weights_only=True)
mel_gan.fit(X_train, y_train, batch_size=8, epochs=10, validation_data=(X_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
# mel_gan.save_weights("C:/Users/Darwin.Cui/OneDrive - University of Southampton/Desktop/SE_dilation/MelGAN/parameters/GAN_mag_pink.h5")



#test step

# def audio_test(noisy_mag, noisy_phase, outpath, SNR):
#     for i in range(len(noisy_mag)):
#         test_original = noisy_mag[i]
#         test_org = np.reshape(test_original, (test_original.shape[1], test_original.shape[0]))
#         phase = noisy_phase[i]
#         test = tf.expand_dims(test_org, axis=0)
#         res = generator.predict(test)
#         res = tf.squeeze(res, axis=0)
#         res = np.reshape(res, (res.shape[1], res.shape[0]))
#         stft = res * np.exp(1j * phase)
#         outcome = librosa.istft(stft, hop_length=255, win_length=510, window='hann', center=True, length=None)
#         write(outpath + 'cleaned' + str(SNR) + '_' + str(i) + '.wav', 16000, outcome)
#
# mag = [obj_noisymag_m2, obj_noisymag_0, obj_noisymag_2, obj_noisymag_4, obj_noisymag_6]
# phase = [obj_phase_m2, obj_phase_0, obj_phase_2, obj_phase_4, obj_phase_6]
# snr = [-2, 0, 2, 4, 6]
# for i in range(len(snr)):
#     audio_test(noisy_mag= mag[i],
#                noisy_phase= phase[i],
#                outpath= 'E:/data/LibriSpeech_100/validation/'+noise_type+'/GAN/noisy_'+str(snr[i])+'/',
#                SNR= snr[i])
# exit()

