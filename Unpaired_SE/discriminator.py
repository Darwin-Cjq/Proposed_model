import tensorflow as tf
from keras import layers
import keras.backend as K
import numpy as np
from joblib import Parallel, delayed
from pesq import pesq



def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


# def batch_pesq(clean, noisy):
#     def _wrapper(clean, noisy):
#         pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
#         pesq_score = np.array(pesq_score)
#         if -1 in pesq_score:
#             return None
#         pesq_score = (pesq_score - 1) / 3.5
#         return pesq_score
#
#     pesq_score = tf.py_function(
#         func=_wrapper,
#         inp=[clean, noisy],
#         Tout=tf.float32
#     )
#     return pesq_score

def batch_pesq(clean, noisy):
    def _wrapper(clean, noisy):
        pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
        pesq_score = np.array(pesq_score)
        if -1 in pesq_score:
            pesq_score = np.zeros_like(pesq_score)
        pesq_score = (pesq_score - 1) / 3.5
        return pesq_score

    pesq_score = tf.py_function(
        func=_wrapper,
        inp=[clean, noisy],
        Tout=tf.float32
    )
    return pesq_score

class Discriminator(tf.keras.Model):
    def __init__(self, ndf):
        super().__init__()
        self.conv1 = layers.Conv2D(ndf, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm1 = layers.BatchNormalization()
        self.prelu1 = layers.PReLU()
        self.conv2 = layers.Conv2D(ndf * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm2 = layers.BatchNormalization()
        self.prelu2 = layers.PReLU()
        self.conv3 = layers.Conv2D(ndf * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm3 = layers.BatchNormalization()
        self.prelu3 = layers.PReLU()
        self.conv4 = layers.Conv2D(ndf * 8, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm4 = layers.BatchNormalization()
        self.prelu4 = layers.PReLU()
        self.pool = layers.GlobalMaxPooling2D()
        self.dense1 = layers.Dense(ndf * 4, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout = layers.Dropout(0.3)
        self.prelu5 = layers.PReLU()
        self.dense2 = layers.Dense(1)
        self.activation = layers.Activation('sigmoid')

    def call(self, x, y):
        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)
        xy = tf.concat([x, y], axis=-1)
        xy = self.conv1(xy)
        xy = self.batchnorm1(xy)
        xy = self.prelu1(xy)
        xy = self.conv2(xy)
        xy = self.batchnorm2(xy)
        xy = self.prelu2(xy)
        xy = self.conv3(xy)
        xy = self.batchnorm3(xy)
        xy = self.prelu3(xy)
        xy = self.conv4(xy)
        xy = self.batchnorm4(xy)
        xy = self.prelu4(xy)
        xy = self.pool(xy)
        xy = self.dense1(xy)
        xy = self.dropout(xy)
        xy = self.prelu5(xy)
        xy = self.dense2(xy)
        xy = self.activation(xy)
        return xy

# 创建一个Discriminator实例
discriminator = Discriminator(ndf=64)


from keras import layers, Model
def create_discriminator(input_shape, ndf):
    x_input = layers.Input(shape=input_shape)
    y_input = layers.Input(shape=input_shape)

    x_input = tf.expand_dims(x_input, axis=-1)
    y_input = tf.expand_dims(y_input, axis=-1)
    xy = layers.Concatenate(axis=-1)([x_input, y_input])

    xy = layers.Conv2D(ndf, (4, 4), strides=(2, 2), padding='same', use_bias=False)(xy)
    xy = layers.BatchNormalization()(xy)
    xy = layers.PReLU()(xy)

    xy = layers.Conv2D(ndf * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False)(xy)
    xy = layers.BatchNormalization()(xy)
    xy = layers.PReLU()(xy)

    xy = layers.Conv2D(ndf * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False)(xy)
    xy = layers.BatchNormalization()(xy)
    xy = layers.PReLU()(xy)

    xy = layers.Conv2D(ndf * 8, (4, 4), strides=(2, 2), padding='same', use_bias=False)(xy)
    xy = layers.BatchNormalization()(xy)
    xy = layers.PReLU()(xy)

    xy = layers.GlobalMaxPooling2D()(xy)

    xy = layers.Dense(ndf * 4, kernel_regularizer=tf.keras.regularizers.l2(0.001))(xy)
    xy = layers.Dropout(0.3)(xy)
    xy = layers.PReLU()(xy)

    xy = layers.Dense(1)(xy)
    xy = layers.Activation('sigmoid')(xy)

    model = Model(inputs=[x_input, y_input], outputs=xy)

    return model


# 使用该函数创建模型
# ndf = 64
# discriminator_model = create_discriminator(input_shape=(256, 256), ndf=ndf)
# x = tf.random.normal((1, 256, 256))
# y = tf.random.normal((1, 256, 256))
# out = discriminator_model([x, y])
# print(out.shape)

