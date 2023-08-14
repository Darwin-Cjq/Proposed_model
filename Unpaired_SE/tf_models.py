import tensorflow as tf
from keras import Model
from keras.layers import Dropout, Conv2D, Conv1D, BatchNormalization, Activation, MultiHeadAttention, Attention, LeakyReLU, Reshape, \
    Conv2DTranspose, AveragePooling1D, PReLU, ELU, LayerNormalization
import keras.backend as K
import Complexnn as complexnn
import numpy as np
import keras
import tensorflow_addons as tfa
import complex_transconv


# the input shape is assumed as (bz, 249, 256)

## frames to wave
def inverse_framing(signal, frame_shift=128):
    return tf.signal.overlap_and_add(signal=signal, frame_step=frame_shift, name=None)

def frame_wave(signal):
    wave_1st = inverse_framing(signal[0])
    wave_1st = tf.expand_dims(wave_1st, axis=0)
    for i in range(1, len(signal)):
        frame = signal[i]
        wave = inverse_framing(frame)
        wave = tf.expand_dims(wave, axis=0)
        wave_1st = tf.concat([wave_1st, wave], axis=0)
    return wave_1st

class SEModule(Model):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.layer_1 = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last', keepdims=True)
        self.layer_2 = tf.keras.layers.Conv1D(bottleneck, kernel_size=1, padding='valid')
        self.layer_3 = tf.keras.layers.ReLU()
        self.layer_4 = tf.keras.layers.LayerNormalization()
        self.layer_5 = tf.keras.layers.Conv1D(channels, kernel_size=1, padding='valid')
        self.layer_6 = tf.keras.layers.Activation('sigmoid')

    def call(self, input):
        # x = self.se(input)
        x = self.layer_1(input)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        return input * x

# model = SEModule(channels=256)
# input_darwin = K.random_uniform(shape= [2, 249, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res1 = model(input_darwin)
# print(res1.shape)
# exit()

def inverse_framing(sig, frame_shift=128):
    return tf.signal.overlap_and_add(signal=sig, frame_step=frame_shift, name=None)

def wave_mel(wave, sample_rate = 16000):
    # A 1024-point STFT with frames of 1024/sample_rate ms and 50% overlap.
    stfts = tf.signal.stft(wave, frame_length=1024, frame_step=512,
                           fft_length=1024)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    return mel_spectrograms



class PositionalEncoding(tf.keras.layers.Layer):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 创建一个足够长的P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)

class Transformer_encoder(tf.keras.Model):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, if_pos_encoding=False):
        super(Transformer_encoder, self).__init__()
        self.if_pos_encoding = if_pos_encoding
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.multi_att = MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)
        self.layer_dropout_1 = Dropout(self.dropout)
        self.LN_1 = LayerNormalization(epsilon=1e-6)
        self.conv1d_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")
        self.layer_dropout_2 = Dropout(self.dropout)
        self.conv1d_2 = Conv1D(filters=512, kernel_size=1)
        self.LN_2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        if self.if_pos_encoding:
            pos_encoding = PositionalEncoding(num_hiddens=inputs.shape[-1], dropout=0)
            positions = pos_encoding(inputs * tf.math.sqrt(tf.cast(inputs.shape[-1], dtype=tf.float32)))
            inputs = inputs + positions

        x = self.multi_att(inputs, inputs)
        x = self.layer_dropout_1(x)
        x = self.LN_1(x)
        res = x + inputs

        x = self.conv1d_1(res)
        x = self.layer_dropout_2(x)
        x = self.conv1d_2(x)
        x = self.LN_2(x)
        result = x + res
        return result

class complex_MHA(tf.keras.Model):
    def __init__(self, key_dim, num_heads, dropout=0.1):
        super(complex_MHA, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.real_multi_att = MultiHeadAttention(key_dim=self.key_dim, num_heads=self.num_heads, dropout=self.dropout)
        self.imag_multi_att = MultiHeadAttention(key_dim=self.key_dim, num_heads=self.num_heads, dropout=self.dropout)

    def call(self, inputs):
        real, imag = tf.split(inputs, 2, axis=-1)
        real = self.real_multi_att(real, real)
        imag = self.imag_multi_att(imag, imag)
        result = tf.concat([real, imag], axis=-1)
        return result

# input_data = K.random_uniform(shape=(2, 100, 512))
# model = complex_MHA(key_dim=512, num_heads=8)
# output_data = model(input_data)
# print(output_data.shape)
# exit()



class complex_transformer(tf.keras.Model):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, if_pos_encoding=False):
        super(complex_transformer, self).__init__()
        self.if_pos_encoding = if_pos_encoding
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.real_multi_att = MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)
        self.real_layer_dropout_1 = Dropout(self.dropout)
        self.real_LN_1 = LayerNormalization(epsilon=1e-6)
        self.real_conv1d_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")
        self.real_layer_dropout_2 = Dropout(self.dropout)
        self.real_conv1d_2 = Conv1D(filters=8, kernel_size=1)
        self.real_LN_2 = LayerNormalization(epsilon=1e-6)

        self.imag_multi_att = MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)
        self.imag_layer_dropout_1 = Dropout(self.dropout)
        self.imag_LN_1 = LayerNormalization(epsilon=1e-6)
        self.imag_conv1d_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")
        self.imag_layer_dropout_2 = Dropout(self.dropout)
        self.imag_conv1d_2 = Conv1D(filters=8, kernel_size=1)
        self.imag_LN_2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        real, imag = tf.split(inputs, 2, axis=-1)
        if self.if_pos_encoding:
            real_pos_encoding = PositionalEncoding(num_hiddens=real.shape[-1], dropout=0)
            real_positions = real_pos_encoding(real * tf.math.sqrt(tf.cast(real.shape[-1], dtype=tf.float32)))
            real = real + real_positions
            imag_pos_encoding = PositionalEncoding(num_hiddens=imag.shape[-1], dropout=0)
            imag_positions = imag_pos_encoding(imag * tf.math.sqrt(tf.cast(imag.shape[-1], dtype=tf.float32)))
            imag = imag + imag_positions

        real_part = self.real_multi_att(real, real)
        real_part = self.real_layer_dropout_1(real_part)
        real_part = self.real_LN_1(real_part)
        real_res = real + real_part

        real_part = self.real_conv1d_1(real_res)
        real_part = self.real_layer_dropout_2(real_part)
        real_part = self.real_conv1d_2(real_part)
        real_part = self.real_LN_2(real_part)
        real_result = real_res + real_part

        imag_part = self.imag_multi_att(imag, imag)
        imag_part = self.imag_layer_dropout_1(imag_part)
        imag_part = self.imag_LN_1(imag_part)
        imag_res = imag + imag_part

        imag_part = self.imag_conv1d_1(imag_res)
        imag_part = self.imag_layer_dropout_2(imag_part)
        imag_part = self.imag_conv1d_2(imag_part)
        imag_part = self.imag_LN_2(imag_part)
        imag_result = imag_res + imag_part

        result = tf.concat([real_result, imag_result], axis=-1)
        return result

# input_data = K.random_uniform(shape=(2, 100, 512))
# model = complex_transformer(head_size=512, num_heads=2, ff_dim=2048, dropout=0.1, if_pos_encoding=False)
# output_data = model(input_data)
# print(output_data.shape)
# exit()


class Generator_speech(tf.keras.Model):
    def __init__(self, kernel_size, strides, feature_timeaxis, transformer=True):
        super(Generator_speech, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.inp_reshape = tf.expand_dims
        self.feature_timeaxis = feature_timeaxis
        self.transformer = transformer

        self.c1 = complexnn.conv.ComplexConv1D(filters=128, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.b1 = complexnn.ComplexLayerNorm()
        self.a1 = ELU(alpha=1.0)

        self.c2 = complexnn.conv.ComplexConv1D(filters=64, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.b2 = complexnn.ComplexLayerNorm()
        self.a2 = ELU(alpha=1.0)

        self.c3 = complexnn.conv.ComplexConv1D(filters=32, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.b3 = complexnn.ComplexLayerNorm()
        self.a3 = ELU(alpha=1.0)

        self.c4 = complexnn.conv.ComplexConv1D(filters=16, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.b4 = complexnn.ComplexLayerNorm()
        self.a4 = ELU(alpha=1.0)

        self.c5 = complexnn.conv.ComplexConv1D(filters=8, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.b5 = complexnn.ComplexLayerNorm()
        self.a5 = ELU(alpha=1.0)
        # the output shape of this layer is (bz, T, 8, 64)

        self.Reshape_1 = Reshape
        self.conv1d_1 = Conv1D(filters=512, kernel_size=1, strides=1, data_format='channels_last', padding='same')

        # the output shape of this layer is (bz, T, 256)
        self.middle_MHSA_1 = complex_MHA(key_dim=128, num_heads=4)
        self.middle_MHSA_2 = complex_MHA(key_dim=128, num_heads=4)
        self.transformer_encoder_1 = complex_transformer(head_size=128, num_heads=4, ff_dim=256)
        self.transformer_encoder_2 = complex_transformer(head_size=128, num_heads=4, ff_dim=256)
        self.Reshape_2 = Reshape

        self.d5 = complex_transconv.ComplexTransposedConv1D(filters=16, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.db5 = complexnn.ComplexLayerNorm()
        self.da5 = ELU(alpha=1.0)

        self.d4 = complex_transconv.ComplexTransposedConv1D(filters=32, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.db4 = complexnn.ComplexLayerNorm()
        self.da4 = ELU(alpha=1.0)

        self.d3 = complex_transconv.ComplexTransposedConv1D(filters=64, kernel_size=self.kernel_size, strides=self.strides, padding='same', output_padding=1)
        self.db3 = complexnn.ComplexLayerNorm()
        self.da3 = ELU(alpha=1.0)

        self.d2 = complex_transconv.ComplexTransposedConv1D(filters=128, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.db2 = complexnn.ComplexLayerNorm()
        self.da2 = ELU(alpha=1.0)

        self.d1 = complex_transconv.ComplexTransposedConv1D(filters=256, kernel_size=self.kernel_size, strides=self.strides, padding='same')
        self.db1 = complexnn.ComplexLayerNorm()
        self.da1 = ELU(alpha=1.0)


    def call(self, inputs):
        # input = self.inp_reshape(input=inputs, axis=-1)
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
        # the output shape of this layer is (bz, 4, 16)

        if self.transformer:
            x_middle1 = self.transformer_encoder_1(x_c5)
            x_middle2 = self.transformer_encoder_2(x_middle1)
        else:
            x_middle1 = self.middle_MHSA_1(x_c5, x_c5)
            x_middle2 = self.middle_MHSA_2(x_middle1, x_middle1)

        # the shape of x is (bz, 4, 16)
        x = tf.concat([x_middle2, x_c5], axis=-1)
        x = self.d5(x)
        x = self.db5(x)
        x_d5 = self.da5(x)

        x = tf.concat([x_d5, x_c4], axis=-1)
        x = self.d4(x)
        x = self.db4(x)
        x_d4 = self.da4(x)

        x = tf.concat([x_d4, x_c3], axis=-1)
        x = self.d3(x)
        x = self.db3(x)
        x_d3 = self.da3(x)

        x = tf.concat([x_d3, x_c2], axis=-1)
        x = self.d2(x)
        x = self.db2(x)
        x_d2 = self.da2(x)

        x = tf.concat([x_d2, x_c1], axis=-1)
        x = self.d1(x)
        x = self.db1(x)
        # x_d1 = self.da1(x)

        return x

encoder_speech = Generator_speech(kernel_size=4, strides=2, feature_timeaxis=124)
# input_darwin = K.random_uniform(shape=[2, 124, 512], minval=0.0, maxval=1.0, dtype=None, seed=None)
# input_darwin = tf.convert_to_tensor(input_darwin)
# res = encoder_speech(input_darwin)
# print(res.shape)
# exit()

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

class Gen_AIA(tf.keras.Model):
    def __init__(self, kernel_size, strides, filters, gate_filter, gate_kernel_1, gate_kernel_2):
        super(Gen_AIA, self).__init__()
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
        self.glu_0 = gate_dilation(filter=self.filters[0], kernel_size_l1=self.kernel_size[0],
                                   kernel_size_l2=self.kernel_size[0], dilation_rate=1)
        self.en_block0 = keras.models.Sequential([self.conv1d_0, self.LN_0, self.act_0, self.glu_0])
        ###
        self.conv1d_1 = complexnn.conv.ComplexConv1D(filters=self.filters[1], kernel_size=self.kernel_size[1],
                                                     strides=self.strides[1],
                                                     padding='same')
        self.LN_1 = complexnn.ComplexLayerNorm()
        self.act_1 = ELU(alpha=1.0)
        self.glu_1 = gate_dilation(filter=self.filters[1], kernel_size_l1=self.kernel_size[1],
                                   kernel_size_l2=self.kernel_size[1], dilation_rate=1)
        self.en_block1 = keras.models.Sequential([self.conv1d_1, self.LN_1, self.act_1, self.glu_1])
        ###
        self.conv1d_2 = complexnn.conv.ComplexConv1D(filters=self.filters[2], kernel_size=self.kernel_size[2],
                                                     strides=self.strides[2],
                                                     padding='same')
        self.LN_2 = complexnn.ComplexLayerNorm()
        self.act_2 = ELU(alpha=1.0)
        self.glu_2 = gate_dilation(filter=self.filters[2], kernel_size_l1=self.kernel_size[2],
                                   kernel_size_l2=self.kernel_size[2], dilation_rate=1)
        self.en_block2 = keras.models.Sequential([self.conv1d_2, self.LN_2, self.act_2, self.glu_2])

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
        self.conv1d_2 = complexnn.conv.ComplexConv1D(filters=32, kernel_size=1, strides=1, padding='same')
        ##the output shape is [batch_size, nfft/2, T, 128]


        # Decoder layers
        self.deconv1d_2 = complex_transconv.ComplexTransposedConv1D(filters=self.filters[2], kernel_size=self.kernel_size[2], strides=self.strides[2],
                               padding='same')
        self.deLN_2 = complexnn.ComplexLayerNorm()
        self.deact_2 = ELU(alpha=1.0)
        self.deglu_2 = gate_dilation(filter=self.filters[2], kernel_size_l1=self.kernel_size[2],
                                   kernel_size_l2=self.kernel_size[2], dilation_rate=1)
        self.de_block2 = keras.models.Sequential([self.deconv1d_2, self.deLN_2, self.deact_2, self.deglu_2])

        ###

        self.deconv1d_1 = complex_transconv.ComplexTransposedConv1D(filters=self.filters[1], kernel_size=self.kernel_size[1],
                                          strides=self.strides[1],
                                          padding='same')
        self.deLN_1 = complexnn.ComplexLayerNorm()
        self.deact_1 = ELU(alpha=1.0)
        self.deglu_1 = gate_dilation(filter=self.filters[1], kernel_size_l1=self.kernel_size[1],
                                     kernel_size_l2=self.kernel_size[1], dilation_rate=1)
        self.de_block1 = keras.models.Sequential([self.deconv1d_1, self.deLN_1, self.deact_1, self.deglu_1])

        ###

        self.deconv1d_0 = complex_transconv.ComplexTransposedConv1D(filters=self.filters[-1], kernel_size=self.kernel_size[0],
                                          strides=self.strides[0],
                                          padding='same')
        self.deact_0 = ELU(alpha=1.0)
        self.de_block0 = keras.models.Sequential([self.deconv1d_0, self.deact_0])

    def call(self, inputs, training=None, mask=None):
        # x = tf.expand_dims(inputs, axis=-1)
        x_0 = self.en_block0(inputs)
        x_1 = self.en_block1(x_0)
        x_2 = self.en_block2(x_1)
        # x_2 = Reshape((x_2.shape[1], x_2.shape[2] * x_2.shape[3]))(x_2)

        x_res1_1, x_jump1_1 = self.g1_1(x_2)
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

        de_feed2 = tf.concat([x_2, total], axis=-1)
        y_2 = self.de_block2(de_feed2)
        de_feed1 = tf.concat([x_1, y_2], axis=-1)
        y_1 = self.de_block1(de_feed1)
        de_feed0 = tf.concat([x_0, y_1], axis=-1)
        y_0 = self.de_block0(de_feed0)
        return y_0


model = Gen_AIA(kernel_size=[3, 4, 4], strides=[1, 2, 2],filters=[128, 64, 32, 256],
                gate_filter=32, gate_kernel_1=5, gate_kernel_2=1)
# input_darwin = K.random_uniform(shape=[2, 124, 512], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res = model(input_darwin)
# print(res.shape)
# exit()

class Gen_model(tf.keras.Model):
    def __init__(self, de_speech, de_noise):
        super(Gen_model, self).__init__()

        self.de_speech = de_speech
        self.de_noise = de_noise

    def call(self, inputs):
        # representation_speech = self.en_speech(inputs)# [batch, 61, 256]
        # x = tf.concat([representation_speech, inputs], axis=1)# [batch, 249, 256]
        # x = self.fusion_layer([inputs, representation_speech])
        # x = representation_speech + inputs
        x = inputs
        speech = self.de_speech(x)
        noise = self.de_noise(x)
        return speech, noise

# model = Gen_model(#en_speech=Speech_Encoder(num_blocks=2, scale=8, mel_freq_size=256),
#                   en_speech=ECAPA_TDNN(C=256, scale=4),
#                   de_speech=Generator_speech(kernel_size=7, strides=(1, 2), feature_timeaxis=249),
#                   de_noise=Gen_aia(kernel_size=[(1,3),(3,5),(3,5)], strides=[(1,1),(1,2),(1,2)],filters=[16,16,64,1]))
model = Gen_model(de_speech=Generator_speech(kernel_size=4, strides=2, feature_timeaxis=124),
                  de_noise=Gen_AIA(kernel_size=[3, 4, 4], strides=[1, 2, 2],filters=[128, 64, 32, 256],
                gate_filter=32, gate_kernel_1=5, gate_kernel_2=1))
# input_darwin = K.random_uniform(shape=[2, 124, 512], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res = model(input_darwin)
# print(res[0].shape, res[1].shape)
# exit()



# input_shape is [none, T, 512]
def discriminator_block(input):
    conv1 = Conv1D(256, 5, 1, "same", groups=256)(input) #output shape is (bz, T, 256)
    lrelu1 = LeakyReLU()(conv1)
    conv2 = Conv1D(128, 5, 2, "same", groups=64)(lrelu1)#output shape is (bz, T/2, 128)
    lrelu2 = LeakyReLU()(conv2)
    conv3 = Conv1D(64, 5, 2, "same", groups=32)(lrelu2)#output shape is (bz, T/4, 64)
    lrelu3 = LeakyReLU()(conv3)
    conv4 = Conv1D(32, 5, 2, "same", groups=16)(lrelu3)#output shape is (bz, T/8, 32)
    lrelu4 = LeakyReLU()(conv4)
    conv5 = Conv1D(16, 5, 2, "same", groups=8)(lrelu4)#output shape is (bz, T/16, 16)
    lrelu5 = LeakyReLU()(conv5)
    conv6 = Conv1D(8, 3, 1, "same")(lrelu5)#output shape is (bz, T/16, 8)
    lrelu6 = LeakyReLU()(conv6)
    conv7 = Conv1D(1, 2, 1, "same")(lrelu6)#output shape is (bz, T/16, 1)
    return [lrelu1, lrelu2, lrelu3, lrelu4, lrelu5, lrelu6, conv7]

# input shape is (none, 346, 256)
def Discriminator_block(input):
    conv1 = Conv1D(128, 7, 1, "same")(input) #output shape is (bz, T, 128)
    lrelu1 = LeakyReLU()(conv1)
    conv2 = Conv1D(64, 7, 2, "same", groups=16)(lrelu1)#output shape is (bz, T/2, 64)
    lrelu2 = LeakyReLU()(conv2)
    conv3 = Conv1D(32, 5, 2, "same", groups=8)(lrelu2)#output shape is (bz, T/4, 32)
    lrelu3 = LeakyReLU()(conv3)
    conv4 = Conv1D(16, 5, 2, "same", groups=4)(lrelu3)#output shape is (bz, T/8, 16)
    lrelu4 = LeakyReLU()(conv4)
    conv5 = Conv1D(8, 3, 2, "same", groups=2)(lrelu4)#output shape is (bz, T/16, 8)
    lrelu5 = LeakyReLU()(conv5)
    conv6 = Conv1D(4, 3, 1, "same")(lrelu5)#output shape is (bz, T/16, 4)
    lrelu6 = LeakyReLU()(conv6)
    conv7 = Conv1D(1, 2, 1, "same")(lrelu6)#output shape is (bz, T/16, 1)
    # return [lrelu1, lrelu2, lrelu3, lrelu4, lrelu5, lrelu6, conv7]
    return conv7

def Discriminator_melblock(input):
    conv1 = Conv1D(32, 7, 1, "same")(input) #output shape is (bz, T, 128)
    lrelu1 = LeakyReLU()(conv1)
    conv2 = Conv1D(16, 7, 2, "same", groups=16)(lrelu1)#output shape is (bz, T/2, 64)
    lrelu2 = LeakyReLU()(conv2)
    conv3 = Conv1D(8, 5, 2, "same", groups=8)(lrelu2)#output shape is (bz, T/4, 32)
    lrelu3 = LeakyReLU()(conv3)
    conv4 = Conv1D(1, 2, 1, "same")(lrelu3)#output shape is (bz, T/16, 1)

    return conv4

def func(wave, sample_rate = 16000):
    # A 1024-point STFT with frames of 1024/sample_rate ms and 50% overlap.
    stfts = tf.signal.stft(wave, frame_length=1024, frame_step=512,
                           fft_length=1024)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    return mel_spectrograms

def Create_discriminator(input_shape, if_mel= True):
    inp = keras.Input(input_shape)

    out_map1 = Discriminator_block(inp)
    pool1 = AveragePooling1D()(inp)
    out_map2 = Discriminator_block(pool1)
    pool2 = AveragePooling1D()(pool1)
    out_map3 = Discriminator_block(pool2)

    if if_mel:
        inp_wave = tf.signal.overlap_and_add(signal=inp, frame_step=128, name=None)#from frame to wave
        inp_mel = func(inp_wave)

        mel_map1 = Discriminator_melblock(inp_mel)
        mel_pool1 = AveragePooling1D()(inp_mel)
        mel_map2 = Discriminator_melblock(mel_pool1)
        mel_pool2 = AveragePooling1D()(mel_pool1)
        mel_map3 = Discriminator_melblock(mel_pool2)
        return keras.Model(inp, [out_map1, out_map2, out_map3, mel_map1, mel_map2, mel_map3])

    else:
        return keras.Model(inp, [out_map1, out_map2, out_map3])


# Discriminator = Create_discriminator((124, 256), if_mel= True)
# input_darwin = K.random_uniform(shape=[2, 124, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res1 = Discriminator(input_darwin)#the shape of out_map3.conv7 is (2, 4, 1)
# print(res1) #(2, 16, 1), (2, 8, 1), (2, 4, 1), (2, 16, 1), (2, 8, 1), (2, 4, 1)
# exit()

def create_discriminator(input_shape):
    inp = keras.Input(input_shape)
    out_map1 = discriminator_block(inp)
    pool1 = AveragePooling1D()(inp)
    out_map2 = discriminator_block(pool1)
    pool2 = AveragePooling1D()(pool1)
    out_map3 = discriminator_block(pool2)
    return keras.Model(inp, [out_map1, out_map2, out_map3])

Discriminator = create_discriminator((124, 256))
# input_darwin = K.random_uniform(shape= [2, 124, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res1 = Discriminator(input_darwin)#the shape of out_map3.conv7 is (2, 8, 1)
# print(res1)
# exit()
