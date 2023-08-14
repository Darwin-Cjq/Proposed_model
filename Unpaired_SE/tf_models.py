import tensorflow as tf
from keras import Model
from keras.layers import Dropout, Conv2D, Conv1D, BatchNormalization, Activation, MultiHeadAttention, Attention, LeakyReLU, AveragePooling2D, \
    Conv2DTranspose, AveragePooling1D, Reshape, ELU, LayerNormalization
import keras.backend as K
import numpy as np
import keras
import tensorflow_addons as tfa
import ATFA
import AHA

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

class Res2net_block(Model):
    def __init__(self, freq_size=256, kernel_size=3, strides=1, dilation=1, scale=4):
        super(Res2net_block, self).__init__()
        self.freq_size = freq_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.scale = scale

        self.conv1 = tf.keras.layers.Conv1D(filters = self.freq_size, kernel_size = 1, strides = self.strides, padding = 'same', dilation_rate = self.dilation)
        self.LN1 = tf.keras.layers.LayerNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.convs = []
        self.LNs = []
        for i in range(self.scale-1):
            self.convs.append(tf.keras.layers.Conv1D(filters = self.freq_size/self.scale, kernel_size = self.kernel_size, strides = self.strides, padding = 'same', dilation_rate = self.dilation))
            self.LNs.append(tf.keras.layers.LayerNormalization())

        self.conv2 = tf.keras.layers.Conv1D(filters = self.freq_size, kernel_size = 1, strides = self.strides, padding = 'same', dilation_rate = self.dilation)
        self.LN2 = tf.keras.layers.LayerNormalization()
        self.SE = SEModule(self.freq_size)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.LN1(x)
        x = self.relu(x)
        x = tf.split(x, self.scale, axis=-1)
        y = x[0]

        for i in range(1, self.scale):
            x[i] = self.convs[i - 1](x[i])
            x[i] = self.LNs[i - 1](x[i])
            x[i] = self.relu(x[i])
            y = tf.concat([y, x[i]], axis=-1)
            if i != self.scale - 1:
                x[i + 1] = x[i + 1] + x[i]

        x = self.conv2(y)
        x = self.LN2(x)
        x = self.relu(x)
        x = self.SE(x)
        out = x + inputs
        return out

# model = Res2net_block(freq_size=256)
# input_darwin = K.random_uniform(shape= [2, 249, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res1 = model(input_darwin)
# print(res1.shape)
# exit()

class ECAPA_TDNN(tf.keras.Model):
    def __init__(self, C=256, mel_freq_size=80, scale=4):
        super(ECAPA_TDNN, self).__init__()
        self.mel_freq_size = mel_freq_size
        self.scale = scale
        self.conv1 = tf.keras.layers.Conv1D(C, kernel_size=5, strides=1, padding='same')
        self.relu = tf.keras.layers.ReLU()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.layer1 = Res2net_block(freq_size=C, scale=self.scale)
        self.layer2 = Res2net_block(freq_size=C, scale=self.scale)
        self.layer3 = Res2net_block(freq_size=C, scale=self.scale)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = tf.keras.layers.Conv1D(filters=C, kernel_size=1)

        self.self_attention_1 = Attention()
        self.self_attention_2 = Attention()
        self.self_attention_3 = Attention()

        self.conv1d_out = tf.keras.layers.Conv1D(filters=C, kernel_size=1)

    def call(self, x):
        # inputs是frames
        # 首先将frames转为wave
        inputs_wave = inverse_framing(x)
        # 之后将wave转为mel
        inputs_mel = wave_mel(inputs_wave)
        x = self.conv1(inputs_mel)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x) #（batch_size，61，256）
        x2 = self.layer2(x+x1) #（batch_size，61，256）
        x3 = self.layer3(x+x1+x2) #（batch_size，61，256）

        x = self.layer4(tf.concat((x1, x2, x3), axis=-1))
        x = self.relu(x) #（batch_size，61，256 * 3）

        x_att_1 = self.self_attention_1([x, x1])
        x_att_2 = self.self_attention_2([x, x2])
        x_att_3 = self.self_attention_3([x, x3])

        x_att = tf.concat((x_att_1, x_att_2, x_att_3), axis=-1)
        out = self.conv1d_out(x_att)

        return out

# model = ECAPA_TDNN(C=256, scale=4)
# input_darwin = K.random_uniform(shape=[2, 249, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res1 = model(input_darwin)#the shape of out_map3.conv7 is (2, 8, 1)
# print(res1.shape)
# exit()


class Speech_Encoder(Model):
    def __init__(self, num_blocks, scale, mel_freq_size):
        super(Speech_Encoder, self).__init__()
        self.num_blocks = num_blocks
        self.scale = scale
        self.mel_freq_size = mel_freq_size
        self.block = []
        for i in range(self.num_blocks):
            self.block.append(Res2net_block(freq_size=self.mel_freq_size, scale=self.scale))
        self.conv1d = tf.keras.layers.Conv1D(filters=self.mel_freq_size, kernel_size = 1, strides = 1, padding = 'same', dilation_rate = 1)

    def call(self, inputs):
        # inputs是frames
        # 首先将frames转为wave
        inputs_wave = inverse_framing(inputs)
        # 之后将wave转为mel
        inputs_mel = wave_mel(inputs_wave)
        out = []
        x = inputs_mel
        for i in range(self.num_blocks):
            x = self.block[i](x)
            out.append(x)
        out = tf.concat(out, axis=-1)
        out = self.conv1d(out)
        return out

# model = Speech_Encoder(num_blocks=2, scale=8, mel_freq_size=256)
# input_darwin = K.random_uniform(shape= [2, 249, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res1 = model(input_darwin)
# print(res1.shape)
# exit()

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
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1):
        super(Transformer_encoder, self).__init__()
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
        # pos_encoding = PositionalEncoding(num_hiddens=inputs.shape[-1], dropout=0)
        # positions = pos_encoding(inputs * tf.math.sqrt(tf.cast(inputs.shape[-1], dtype=tf.float32)))
        # inputs = inputs + positions
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


class Generator_speech(tf.keras.Model):
    def __init__(self, kernel_size, strides, feature_timeaxis, transformer=True):
        super(Generator_speech, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.inp_reshape = tf.expand_dims
        self.feature_timeaxis = feature_timeaxis
        self.transformer = transformer

        self.c1 = Conv2D(filters=4, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                         data_format='channels_last')
        self.b1 = LayerNormalization()
        self.a1 = ELU(alpha=1.0)

        self.c2 = Conv2D(filters=8, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                         data_format='channels_last')
        self.b2 = LayerNormalization()
        self.a2 = ELU(alpha=1.0)

        self.c3 = Conv2D(filters=16, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                         data_format='channels_last')
        self.b3 = LayerNormalization()
        self.a3 = ELU(alpha=1.0)

        self.c4 = Conv2D(filters=32, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                         data_format='channels_last')
        self.b4 = LayerNormalization()
        self.a4 = ELU(alpha=1.0)

        self.c5 = Conv2D(filters=64, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                         data_format='channels_last')
        self.b5 = LayerNormalization()
        self.a5 = ELU(alpha=1.0)
        # the output shape of this layer is (bz, T, 8, 64)

        self.Reshape_1 = Reshape
        self.conv1d_1 = Conv1D(filters=512, kernel_size=1, strides=1, data_format='channels_last', padding='same')

        # the output shape of this layer is (bz, T, 256)
        self.middle_MHSA_1 = MultiHeadAttention(num_heads=2, key_dim=512)
        self.middle_MHSA_2 = MultiHeadAttention(num_heads=2, key_dim=1024)
        self.transformer_encoder_1 = Transformer_encoder(head_size=512, num_heads=2, ff_dim=256*2)
        self.transformer_encoder_2 = Transformer_encoder(head_size=512, num_heads=2, ff_dim=256*2)
        self.Reshape_2 = Reshape

        self.d5 = Conv2DTranspose(filters=32, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                  data_format='channels_last')
        self.db5 = LayerNormalization()
        self.da5 = ELU(alpha=1.0)
        self.d4 = Conv2DTranspose(filters=16, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                  data_format='channels_last')
        self.db4 = LayerNormalization()
        self.da4 = ELU(alpha=1.0)
        self.d3 = Conv2DTranspose(filters=8, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                  data_format='channels_last')
        self.db3 = LayerNormalization()
        self.da3 = ELU(alpha=1.0)
        self.d2 = Conv2DTranspose(filters=4, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                  data_format='channels_last')
        self.db2 = LayerNormalization()
        self.da2 = ELU(alpha=1.0)
        self.d1 = Conv2DTranspose(filters=1, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                                  data_format='channels_last')
        self.db1 = LayerNormalization()
        self.da1 = tf.keras.layers.LeakyReLU()
        # self.da1 = Activation('linear')
        # self.da1 = Activation('tanh')
        self.Reshape_3 = tf.squeeze
        self.conv1d_out = Conv1D(filters=256, kernel_size=3, strides=1, padding='same')
        self.a6 = tf.keras.layers.LeakyReLU()

    def call(self, inputs):
        input = self.inp_reshape(input=inputs, axis=-1)
        x = self.c1(input)
        x = self.b1(x)
        x_c1 = self.a1(x)
        # (bz, 249, 128, 4)
        x = self.c2(x_c1)
        x = self.b2(x)
        x_c2 = self.a2(x)
        # (bz, 249, 64, 8)
        x = self.c3(x_c2)
        x = self.b3(x)
        x_c3 = self.a3(x)
        # (bz, 249, 64, 16)

        reshape_xc3 = self.Reshape_1((self.feature_timeaxis, 32 * 16))(x_c3)
        # reshape_xc3 = self.conv1d_1(reshape_xc3)

        # the output shape of this layer is (bz, T, 1024)
        if self.transformer:
            x_middle1 = self.transformer_encoder_1(reshape_xc3)
            x_middle2 = self.transformer_encoder_2(x_middle1)
        else:
            x_middle1 = self.middle_MHSA_1(reshape_xc3, reshape_xc3)
            x_middle2 = self.middle_MHSA_2(x_middle1, x_middle1)

        x = self.Reshape_2((self.feature_timeaxis, 32, 16))(x_middle2)
        # the shape of x is (bz, T, 8, 64)

        x = tf.concat([x, x_c3], axis=-1)
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
        x_d1 = self.da1(x)

        output_matrix = self.Reshape_3(x_d1, axis=-1)
        output_matrix = self.conv1d_out(output_matrix)
        output_matrix = self.a6(output_matrix)

        return output_matrix

# encoder_speech = Generator_speech(kernel_size=7, strides=(1, 2), feature_timeaxis=249)
# input_darwin = K.random_uniform(shape=[2, 249, 512], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res = encoder_speech(input_darwin)
# print(res.shape)
# exit()

class gate_dilation(tf.keras.Model):
    def __init__(self, filter, kernel_size_l1, kernel_size_l2, dilation_rate, data_format='channels_last', padding='same'):
        super(gate_dilation, self).__init__()
        self.filter = filter
        self.kernel_size_l1 = kernel_size_l1
        self.kernel_size_l2 = kernel_size_l2
        self.dilation_rate = dilation_rate
        self.data_format = data_format
        self.padding = padding
        self.a1_1 = Activation('linear')
        self.a1_2 = Activation('sigmoid')
        self.d1_1 = Conv2D(filters=self.filter, kernel_size=self.kernel_size_l1, strides=1,
                           dilation_rate=self.dilation_rate, data_format=self.data_format, padding= self.padding)
        self.d1_2 = Conv2D(filters=self.filter, kernel_size=self.kernel_size_l1, strides=1,
                           dilation_rate=self.dilation_rate, data_format=self.data_format, padding= self.padding)
        self.product_1 = tf.multiply

    def call(self, inputs):
        x1_1 = self.d1_1(inputs)
        x1_2 = self.d1_2(inputs)
        ax1_1 = self.a1_1(x1_1)
        ax1_2 = self.a1_2(x1_2)
        outcome = self.product_1(ax1_2, ax1_1)

        return outcome

class Gen_aia(tf.keras.Model):
    def __init__(self, kernel_size, strides, filters):
        super(Gen_aia, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        # self.glu_kernel_1 = glu_kernel_1
        # self.glu_kernel_2 = glu_kernel_2

        # Encoder layers
        self.conv2d_0 = Conv2D(filters=self.filters[0], kernel_size=self.kernel_size[0], strides=self.strides[0], padding='same',
                         data_format='channels_last')
        self.glu_0 = gate_dilation(filter=self.filters[0], kernel_size_l1=self.kernel_size[0], kernel_size_l2=self.kernel_size[0], dilation_rate=1)
        self.en_block0 = keras.models.Sequential([self.conv2d_0, self.glu_0])
        ###
        self.conv2d_1 = Conv2D(filters=self.filters[1], kernel_size=self.kernel_size[1], strides=self.strides[1],
                               padding='same',
                               data_format='channels_last')
        self.LN_1 = LayerNormalization()
        self.act_1 = ELU(alpha=1.0)
        self.glu_1 = gate_dilation(filter=self.filters[1], kernel_size_l1=self.kernel_size[1],
                                   kernel_size_l2=self.kernel_size[1], dilation_rate=1)
        self.en_block1 = keras.models.Sequential([self.conv2d_1, self.LN_1, self.act_1, self.glu_1])
        ###
        self.conv2d_2 = Conv2D(filters=self.filters[2], kernel_size=self.kernel_size[2], strides=self.strides[2],
                               padding='same',
                               data_format='channels_last')
        self.LN_2 = LayerNormalization()
        self.act_2 = ELU(alpha=1.0)
        self.glu_2 = gate_dilation(filter=self.filters[2], kernel_size_l1=self.kernel_size[2],
                                   kernel_size_l2=self.kernel_size[2], dilation_rate=1)
        self.en_block2 = keras.models.Sequential([self.conv2d_2, self.LN_2, self.act_2, self.glu_2])

        # ATFA blocks
        self.ATFA_0 = ATFA.model
        self.ATFA_1 = ATFA.model
        self.ATFA_2 = ATFA.model

        # AHA blocks
        self.AHA = AHA.model

        # Decoder layers
        self.deconv2d_2 = Conv2DTranspose(filters=self.filters[2], kernel_size=self.kernel_size[2], strides=self.strides[2],
                               padding='same',
                               data_format='channels_last')
        self.deLN_2 = LayerNormalization()
        self.deact_2 = ELU(alpha=1.0)
        self.deglu_2 = gate_dilation(filter=self.filters[2], kernel_size_l1=self.kernel_size[2],
                                   kernel_size_l2=self.kernel_size[2], dilation_rate=1)
        self.de_block2 = keras.models.Sequential([self.deconv2d_2, self.deLN_2, self.deact_2, self.deglu_2])

        ###

        self.deconv2d_1 = Conv2DTranspose(filters=self.filters[1], kernel_size=self.kernel_size[1],
                                          strides=self.strides[1],
                                          padding='same',
                                          data_format='channels_last')
        self.deLN_1 = LayerNormalization()
        self.deact_1 = ELU(alpha=1.0)
        self.deglu_1 = gate_dilation(filter=self.filters[1], kernel_size_l1=self.kernel_size[1],
                                     kernel_size_l2=self.kernel_size[1], dilation_rate=1)
        self.de_block1 = keras.models.Sequential([self.deconv2d_1, self.deLN_1, self.deact_1, self.deglu_1])

        ###

        self.deconv2d_0 = Conv2DTranspose(filters=self.filters[-1], kernel_size=self.kernel_size[0],
                                          strides=self.strides[0],
                                          padding='same',
                                          data_format='channels_last')
        self.deact_0 = Activation('tanh')
        self.de_block0 = keras.models.Sequential([self.deconv2d_0, self.deact_0])

        self.conv_out = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last')

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, axis=-1)
        x_0 = self.en_block0(x)
        x_1 = self.en_block1(x_0)
        x_2 = self.en_block2(x_1)

        atfa_0 = self.ATFA_0(x_2)
        atfa_1 = self.ATFA_0(atfa_0)
        atfa_2 = self.ATFA_0(atfa_1)
        Atfa_0 = tf.expand_dims(atfa_0, axis=-1)
        Atfa_1 = tf.expand_dims(atfa_1, axis=-1)
        Atfa_2 = tf.expand_dims(atfa_2, axis=-1)
        Atfa = tf.concat([Atfa_0, Atfa_1, Atfa_2], axis=-1)

        Aha = self.AHA(Atfa)
        Aha = Aha + tf.squeeze(Atfa_2, axis=-1)

        de_feed2 = tf.concat([x_2, Aha], axis=-1)
        y_2 = self.de_block2(de_feed2)
        de_feed1 = tf.concat([x_1, y_2], axis=-1)
        y_1 = self.de_block1(de_feed1)
        de_feed0 = tf.concat([x_0, y_1], axis=-1)
        y_0 = self.de_block0(de_feed0)

        output_matrix = tf.squeeze(y_0, axis=-1)
        output_matrix = self.conv_out(output_matrix)
        return output_matrix

# model = Gen_aia(kernel_size=[(1,3),(3,5),(3,5)], strides=[(1,1),(1,2),(1,2)],filters=[16,16,64,1])
# input_darwin = K.random_uniform(shape= [2, 249, 512], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res = model(input_darwin)
# print(res.shape)
# exit()

class gate_middle(tf.keras.Model):
    def __init__(self, filter, kernel_size_l1, kernel_size_l2, dilation_rate, data_format, padding):
        super(gate_middle, self).__init__()
        self.filter = filter
        self.kernel_size_l1 = kernel_size_l1
        self.kernel_size_l2 = kernel_size_l2
        self.dilation_rate = dilation_rate
        self.data_format = data_format
        self.padding = padding
        self.a1_1 = ELU(alpha=1.0)
        self.a1_2 = Activation('sigmoid')

        #complex diconv2d
        self.d1_1 = Conv2D(filters=self.filter, kernel_size=self.kernel_size_l1, strides=1,
                           dilation_rate=self.dilation_rate, data_format=self.data_format, padding= self.padding)

        #complex diconv2d
        self.d1_2 = Conv2D(filters=self.filter, kernel_size=self.kernel_size_l1, strides=1,
                           dilation_rate=self.dilation_rate, data_format=self.data_format, padding= self.padding)

        self.d2_1 = Conv2D(filters=self.filter, kernel_size=self.kernel_size_l2, strides=1,
                           dilation_rate=self.dilation_rate, data_format=self.data_format, padding= self.padding)
        self.d2_2 = Conv2D(filters=self.filter, kernel_size=self.kernel_size_l2, strides=1,
                           dilation_rate=self.dilation_rate, data_format=self.data_format, padding= self.padding)
        self.b2_1 = LayerNormalization()
        self.b2_2 = LayerNormalization()
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
        self.conv2d_0 = Conv2D(filters=self.filters[0], kernel_size=self.kernel_size[0], strides=self.strides[0], padding='same',
                         data_format='channels_last')
        self.glu_0 = gate_dilation(filter=self.filters[0], kernel_size_l1=self.kernel_size[0], kernel_size_l2=self.kernel_size[0], dilation_rate=1)
        self.en_block0 = keras.models.Sequential([self.conv2d_0, self.glu_0])
        ###
        self.conv2d_1 = Conv2D(filters=self.filters[1], kernel_size=self.kernel_size[1], strides=self.strides[1],
                               padding='same',
                               data_format='channels_last')
        self.LN_1 = LayerNormalization()
        self.act_1 = ELU(alpha=1.0)
        self.glu_1 = gate_dilation(filter=self.filters[1], kernel_size_l1=self.kernel_size[1],
                                   kernel_size_l2=self.kernel_size[1], dilation_rate=1)
        self.en_block1 = keras.models.Sequential([self.conv2d_1, self.LN_1, self.act_1, self.glu_1])
        ###
        self.conv2d_2 = Conv2D(filters=self.filters[2], kernel_size=self.kernel_size[2], strides=self.strides[2],
                               padding='same',
                               data_format='channels_last')
        self.LN_2 = LayerNormalization()
        self.act_2 = ELU(alpha=1.0)
        self.glu_2 = gate_dilation(filter=self.filters[2], kernel_size_l1=self.kernel_size[2],
                                   kernel_size_l2=self.kernel_size[2], dilation_rate=1)
        self.en_block2 = keras.models.Sequential([self.conv2d_2, self.LN_2, self.act_2, self.glu_2])

        # GLU units
        # block 1


        self.g1_1 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=1,
                                  data_format='channels_last', padding='same')
        self.g1_2 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=2,
                                  data_format='channels_last', padding='same')
        self.g1_3 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=4,
                                  data_format='channels_last', padding='same')
        self.g1_4 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=8,
                                  data_format='channels_last', padding='same')
        self.g1_5 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=16,
                                  data_format='channels_last', padding='same')

        # block 2
        self.g2_1 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=1,
                                  data_format='channels_last', padding='same')
        self.g2_2 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=2,
                                  data_format='channels_last', padding='same')
        self.g2_3 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=4,
                                  data_format='channels_last', padding='same')
        self.g2_4 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=8,
                                  data_format='channels_last', padding='same')
        self.g2_5 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=16,
                                  data_format='channels_last', padding='same')

        # block 3
        self.g3_1 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=1,
                                  data_format='channels_last', padding='same')
        self.g3_2 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=2,
                                  data_format='channels_last', padding='same')
        self.g3_3 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=4,
                                  data_format='channels_last', padding='same')
        self.g3_4 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=8,
                                  data_format='channels_last', padding='same')
        self.g3_5 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=16,
                                  data_format='channels_last', padding='same')

        # block 4
        self.g4_1 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=1,
                                  data_format='channels_last', padding='same')
        self.g4_2 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=2,
                                  data_format='channels_last', padding='same')
        self.g4_3 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=4,
                                  data_format='channels_last', padding='same')
        self.g4_4 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=8,
                                  data_format='channels_last', padding='same')
        self.g4_5 = gate_middle(filter=self.gate_filter, kernel_size_l1=self.gate_kernel_1,
                                  kernel_size_l2=self.gate_kernel_2, dilation_rate=16,
                                  data_format='channels_last', padding='same')

        self.m1_conv2d = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', data_format='channels_last')

        self.m2_conv2d = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', data_format='channels_last')

        self.m3_conv2d = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', data_format='channels_last')

        self.m4_conv2d = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', data_format='channels_last')

        self.m5_conv2d = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', data_format='channels_last')
        self.conv2d_2 = Conv2D(filters=64, kernel_size=1, strides=1, data_format='channels_last', padding='same')
        ##the output shape is [batch_size, nfft/2, T, 128]


        # Decoder layers
        self.deconv2d_2 = Conv2DTranspose(filters=self.filters[2], kernel_size=self.kernel_size[2], strides=self.strides[2],
                               padding='same',
                               data_format='channels_last')
        self.deLN_2 = LayerNormalization()
        self.deact_2 = ELU(alpha=1.0)
        self.deglu_2 = gate_dilation(filter=self.filters[2], kernel_size_l1=self.kernel_size[2],
                                   kernel_size_l2=self.kernel_size[2], dilation_rate=1)
        self.de_block2 = keras.models.Sequential([self.deconv2d_2, self.deLN_2, self.deact_2, self.deglu_2])

        ###

        self.deconv2d_1 = Conv2DTranspose(filters=self.filters[1], kernel_size=self.kernel_size[1],
                                          strides=self.strides[1],
                                          padding='same',
                                          data_format='channels_last')
        self.deLN_1 = LayerNormalization()
        self.deact_1 = ELU(alpha=1.0)
        self.deglu_1 = gate_dilation(filter=self.filters[1], kernel_size_l1=self.kernel_size[1],
                                     kernel_size_l2=self.kernel_size[1], dilation_rate=1)
        self.de_block1 = keras.models.Sequential([self.deconv2d_1, self.deLN_1, self.deact_1, self.deglu_1])

        ###

        self.deconv2d_0 = Conv2DTranspose(filters=self.filters[-1], kernel_size=self.kernel_size[0],
                                          strides=self.strides[0],
                                          padding='same',
                                          data_format='channels_last')
        self.deact_0 = tf.keras.layers.LeakyReLU()
        self.de_block0 = keras.models.Sequential([self.deconv2d_0, self.deact_0])

        self.conv_out = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last')

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, axis=-1)
        x_0 = self.en_block0(x)
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
        total_1 = self.m1_conv2d(total_1)

        x_res2_1, x_jump2_1 = self.g2_1(x_res1_5)
        x_res2_2, x_jump2_2 = self.g2_2(x_res2_1)
        x_res2_3, x_jump2_3 = self.g2_3(x_res2_2)
        x_res2_4, x_jump2_4 = self.g2_4(x_res2_3)
        x_res2_5, x_jump2_5 = self.g2_5(x_res2_4)

        total_2 = tf.concat([x_jump2_1, x_jump2_2], axis=-1)
        total_2 = tf.concat([total_2, x_jump2_3], axis=-1)
        total_2 = tf.concat([total_2, x_jump2_4], axis=-1)
        total_2 = tf.concat([total_2, x_jump2_5], axis=-1)
        total_2 = self.m2_conv2d(total_2)

        x_res3_1, x_jump3_1 = self.g3_1(x_res2_5)
        x_res3_2, x_jump3_2 = self.g3_2(x_res3_1)
        x_res3_3, x_jump3_3 = self.g3_3(x_res3_2)
        x_res3_4, x_jump3_4 = self.g3_4(x_res3_3)
        x_res3_5, x_jump3_5 = self.g3_5(x_res3_4)

        total_3 = tf.concat([x_jump3_1, x_jump3_2], axis=-1)
        total_3 = tf.concat([total_3, x_jump3_3], axis=-1)
        total_3 = tf.concat([total_3, x_jump3_4], axis=-1)
        total_3 = tf.concat([total_3, x_jump3_5], axis=-1)
        total_3 = self.m3_conv2d(total_3)

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
        x = self.conv2d_2(total_123)
        total = x_res3_5 + x
        # total = self.product_total([total_1234, x_res1_5, x_res2_5, x_res3_5, x_res4_5], axis=-1)
        # total = self.m5_conv1d(total)

        de_feed2 = tf.concat([x_2, total], axis=-1)
        y_2 = self.de_block2(de_feed2)
        de_feed1 = tf.concat([x_1, y_2], axis=-1)
        y_1 = self.de_block1(de_feed1)
        de_feed0 = tf.concat([x_0, y_1], axis=-1)
        y_0 = self.de_block0(de_feed0)

        output_matrix = tf.squeeze(y_0, axis=-1)
        # output_matrix = self.conv_out(output_matrix)
        return output_matrix

model = Gen_AIA(kernel_size=[(1,3),(3,5),(3,5)], strides=[(1,1),(1,2),(1,2)],filters=[16,16,64,1],
                gate_filter=64, gate_kernel_1=5, gate_kernel_2=1)
# input_darwin = K.random_uniform(shape=[2, 249, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res = model(input_darwin)
# print(res.shape)
# exit()

class Gen_model(tf.keras.Model):
    def __init__(self, en_speech, de_speech, de_noise):
        super(Gen_model, self).__init__()
        self.en_speech = en_speech
        self.fusion_layer = Attention()
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
model = Gen_model(#en_speech=Speech_Encoder(num_blocks=2, scale=8, mel_freq_size=256),
                  en_speech=ECAPA_TDNN(C=256, scale=4),
                  de_speech=Generator_speech(kernel_size=7, strides=(1, 2), feature_timeaxis=249),
                  de_noise=Gen_AIA(kernel_size=[(1,3),(3,5),(3,5)], strides=[(1,1),(1,2),(1,2)],filters=[16,16,64,1],
                gate_filter=64, gate_kernel_1=5, gate_kernel_2=1))
# input_darwin = K.random_uniform(shape=[2, 249, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
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


# Discriminator = Create_discriminator((249, 256), if_mel= True)
# input_darwin = K.random_uniform(shape=[2, 249, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
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

# Discriminator = create_discriminator((249, 256))
# input_darwin = K.random_uniform(shape= [2, 249, 256], minval=0.0, maxval=1.0, dtype=None, seed=None)
# res1 = Discriminator(input_darwin)#the shape of out_map3.conv7 is (2, 8, 1)
# print(res1)
# exit()
