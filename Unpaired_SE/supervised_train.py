#!/mainfs/home/jc18n17/anaconda3/envs/TF2.6/bin/python3.6
import tensorflow as tf
from keras import Model
import keras.backend as K
import numpy as np
from tensorflow import keras
from tensorflow_addons.optimizers import AdamW
import os
import tf_models
import evaluation
import discriminator
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import glu_unit
import tf_train_1 as tf_train

# define generators and corresponding discriminators
inputTime_size = 249
inputFreq_size = 256
epochs = 10
Batch_size = 2
end = 40000
start = 0

generator = tf_models.Gen_model(de_speech=tf_models.Generator_speech(kernel_size=4, strides=2, feature_timeaxis=124),
                                de_noise=tf_models.Gen_AIA(kernel_size=[3, 4, 4], strides=[1, 2, 2], filters=[128, 64, 32, 256],
                                                           gate_filter=32, gate_kernel_1=5, gate_kernel_2=1))
pre_train_model = tf_train.this_model

# define their own losses
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
cosine_loss = tf.keras.losses.CosineSimilarity(axis=-1)

## frames to wave
def inverse_framing(sig, frame_shift=128):
    return tf.signal.overlap_and_add(signal=sig, frame_step=frame_shift, name=None)


def func_mel(wave, sample_rate=16000):
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


def multi_resolution_stft_loss(y_true, y_pred, window_sizes=[256, 512, 1024, 2048], weights=[1, 0.5, 0.25, 0.125],
                               if_frames=False, mean=None, std=None):
    """
    Calculates the multi-resolution STFT loss between two audio signals.

    Args:
        y_true: The true audio signal.
        y_pred: The predicted audio signal.
        window_sizes: A list of window sizes to use for the STFT calculation.
        weights: A list of weights to use for each window size.

    Returns:
        The multi-resolution STFT loss.
    """
    if mean is not None and std is not None:
        y_true = y_true * std + mean
        y_pred = y_pred * std + mean

    if if_frames:  # if frames, then convert framing to wave
        y_true = inverse_framing(y_true)
        y_pred = inverse_framing(y_pred)

    diff_sisdr = evaluation.tf_si_sdr(reference=y_true, estimation=y_pred)
    return diff_sisdr
    # loss = 0
    # for i, window_size in enumerate(window_sizes):
    #     stft_true = tf.signal.stft(y_true, frame_length=window_size, frame_step=window_size // 2,
    #                                fft_length=window_size)
    #     stft_pred = tf.signal.stft(y_pred, frame_length=window_size, frame_step=window_size // 2,
    #                                fft_length=window_size)
    #     mag_true = tf.abs(stft_true)
    #     mag_pred = tf.abs(stft_pred)
    #     # diff = tf.abs(stft_true - stft_pred)
    #     # diff_mag = tf.math.log(mag_true + 1e-6) - tf.math.log(mag_pred + 1e-6)
    #     # loss += weights[i] * tf.reduce_mean(diff)
    #     diff_mag = tf.keras.losses.MeanAbsoluteError()(y_true=mag_true, y_pred=mag_pred)
    #     loss += weights[i] * diff_mag + weights[i] * diff_sisdr
    # return loss + diff_sisdr

def generator_loss(label, fake_pred):
    return mse(label, fake_pred)

def mae_pred(y_true, y_pred):
    return K.abs(K.mean(y_true - y_pred))

def mse_pred(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

def loss_func(y_true, y_pred, alpha=0.5, beta=0.5, gamma=0, input_mean=None, input_std=None):
    # min_length = tf.minimum(tf.shape(y_true)[-1], tf.shape(y_pred)[-1])
    # y_true = y_true[:, :min_length]
    # y_pred = y_pred[:, :min_length]

    loss_sisdr = multi_resolution_stft_loss(y_true=y_true, y_pred=y_pred,
                                            window_sizes=[256, 512, 1024, 2048], weights=[1, 0.5, 0.25, 0.125],
                                            if_frames=False, mean=input_mean, std=input_std)

    loss_mae = mae_pred(y_true=y_true, y_pred=y_pred)
    loss_mse = 0

    # loss_mse = mse_pred(y_true=y_true, y_pred=y_pred)
    return alpha * loss_sisdr + beta * loss_mae + gamma * loss_mse


def reconstruction_loss(y_true, y_pred, if_cosine=False):
    if if_cosine:
        loss_cosine = cosine_loss(y_true=y_true, y_pred=y_pred)
        return mse_pred(y_pred=y_true, y_true=y_pred) + loss_cosine
    else:
        return mse_pred(y_pred=y_true, y_true=y_pred)


def identity_loss(y_true, y_pred, if_cosine=False):  # 加入余弦相似度来判定输出的语音跟目标的clean speech的相似度

    if if_cosine:
        loss_cosine = cosine_loss(y_true=y_true, y_pred=y_pred)
        return mse_pred(y_pred=y_pred[0], y_true=y_true) + loss_cosine
    else:
        return mse_pred(y_pred=y_pred[0], y_true=y_true)


def Identity_loss(y_true, y_pred, input_mean=None, input_std=None, if_frame=False, if_cosine=False):
    # if the input feature is raw waveform
    # wave_loss & mel_spec loss
    pred_outcome = y_pred
    true_outcome = y_true
    wave_loss = mse_pred(y_pred=pred_outcome, y_true=true_outcome)

    if input_mean is not None and input_std is not None:
        pred_outcome = pred_outcome * input_std + input_mean
        true_outcome = true_outcome * input_std + input_mean

    if if_frame:
        pred_wave = inverse_framing(pred_outcome)
        true_wave = inverse_framing(true_outcome)
    elif not if_frame:
        pred_wave = pred_outcome
        true_wave = true_outcome

    pred_mel = func_mel(pred_wave)
    true_mel = func_mel(true_wave)

    mel_loss = 0.01 * mse_pred(y_true=true_mel, y_pred=pred_mel)

    if if_cosine:  # 加入余弦相似度来判定输出的语音跟目标的clean speech的相似度
        loss_cosine = cosine_loss(y_pred=pred_mel, y_true=true_mel)
        mel_loss = mel_loss + loss_cosine

    return wave_loss * 0.99 + mel_loss

def complex_wave(x, frame_length=510, frame_step=255, fft_length=510):
    real, imag = tf.split(x, 2, axis=-1)
    x_complex = tf.complex(real, imag)
    x_wave = tf.signal.inverse_stft(x_complex, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    return x_wave

def complex_mag(x, frame_length=510, frame_step=255, fft_length=510):
    real, imag = tf.split(x, 2, axis=-1)
    x_complex = tf.complex(real, imag)
    x_mag = tf.abs(x_complex)
    return x_mag

def wave_complex(x, frame_length=510, frame_step=255, fft_length=510):
    x_complex = tf.signal.stft(x, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    real = tf.math.real(x_complex)
    imag = tf.math.imag(x_complex)
    x_wave = tf.concat([real, imag], axis=-1)
    return x_wave

def wave_mag(x, frame_length=510, frame_step=255, fft_length=510):
    x_complex = tf.signal.stft(x, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    x_mag = tf.abs(x_complex)
    return x_mag

class Jianqiao_GAN(keras.Model):
    def __init__(self, generator, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator

    def new_compile(self, gen_optimizer,
                    speech_loss,
                    noisy_loss):
        super().compile()

        # Optimizers
        self.gen_optimizer = gen_optimizer
        # Losses
        self.speech_loss = speech_loss
        self.noisy_loss = noisy_loss
        # Trackers
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.speech_loss_tracker = keras.metrics.Mean(name="speech_loss")
        self.noisy_loss_tracker = keras.metrics.Mean(name="noisy_loss")

    def train_step(self, batch, training=True):
        x_batch_train, y_batch_train = batch
        noisy_speech = x_batch_train
        clean_speech = y_batch_train
        complex_clean_speech = wave_complex(clean_speech)
        complex_noisy_speech = wave_complex(noisy_speech)

        with tf.GradientTape() as gen_tape:
            out_gen_noisy = self.generator(complex_noisy_speech)
            gen_noisy_speech = out_gen_noisy[0]
            gen_noisy_noise = out_gen_noisy[1]
            audio_gen_noisy_speech = complex_wave(gen_noisy_speech, frame_length=510, frame_step=255, fft_length=510)
            # audio_gen_noisy_noise = complex_wave(gen_noisy_noise, frame_length=510, frame_step=255, fft_length=510)
            # audio_gen_noisy_noisy = audio_gen_noisy_noise + audio_gen_noisy_speech

            # Generator_loss=loss_func  return alpha * loss_sisdr + beta * loss_mae + gamma * loss_mse
            min_length = 31875
            speech_loss = self.speech_loss(y_true=clean_speech[:, :min_length], y_pred=audio_gen_noisy_speech[:, :min_length], alpha=0.8, beta=0.1,
                                                  gamma=0.1, input_mean=None, input_std=None) + \
                            self.noisy_loss(y_true=complex_clean_speech, y_pred=gen_noisy_speech)
            # noisy_loss = self.noisy_loss(y_true=noisy_speech[:, :min_length], y_pred=audio_gen_noisy_noisy[:, :min_length])

            gen_loss = speech_loss

            grads_gen = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
            self.gen_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_weights))
            self.gen_loss_tracker.update_state(gen_loss)
            self.speech_loss_tracker.update_state(speech_loss)
            # self.noisy_loss_tracker.update_state(noisy_loss)

            return {
                "gen_loss": self.gen_loss_tracker.result(),
                "speech_loss": self.speech_loss_tracker.result(),
                # "noisy_loss": self.noisy_loss_tracker.result()
            }
        # implement the call method

    def call(self, inputs, *args, **kwargs):
        # pass
        return self.generator(wave_complex(inputs))

finetune_model = Jianqiao_GAN(generator=generator)

checkpoint_second_path = "./finetune_param_1/model.ckpt"
if os.path.exists(checkpoint_second_path + '.index'):
    print('-------------load the supervised training model-----------------')
    finetune_model.load_weights(checkpoint_second_path)

# else:
#     checkpoint_first_path = "./pre_train_param_1/model.ckpt"
#     if os.path.exists(checkpoint_first_path + '.index'):
#         print('-------------load the pre_trained model-----------------')
#         pre_train_model.load_weights(checkpoint_first_path)
#         pre_trained_generator = pre_train_model.generator
#         finetune_model=Jianqiao_GAN(generator=pre_trained_generator)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_second_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

def read_audio(audio_path):
    audio_binary = tf.io.read_file(audio_path)
    waveform, sample_rate = tf.audio.decode_wav(audio_binary)
    return waveform, sample_rate

def waveform_norm(waveform, norm_factor=None):
    #waveform shape: [time, sr * time]
    if norm_factor is None:
        c = tf.sqrt(tf.cast(tf.shape(waveform)[-1], tf.float32) / tf.reduce_sum(waveform ** 2.0, axis=-1))
        waveform = tf.transpose(waveform, perm=[1, 0])
        waveform = tf.transpose(waveform * c, perm=[1, 0])
        return waveform, c
    else:
        waveform = tf.transpose(waveform, perm=[1, 0])
        waveform = tf.transpose(waveform * norm_factor, perm=[1, 0])
        return waveform

def func_audio(audio_path, time_length=2, if_norm=False,
               out_type='waveform', power_compress=True,
               norm_factor=None, return_norm=True):
    """
    :param audio_path: 读取的音频路径
    :param time_length: 音频长度，单位为秒，如果音频长度小于time_length，则进行padding，否则进行截断
    :param if_norm: 是否进行归一化
    :param out_type: 输出的特征类型，可选为'waveform'、'spectrogram'、'mel_spectrogram'、'mfcc'
    :param power_compress: 是否进行power_compress
    :param norm_factor: 传入的归一化系数
    :param return_norm: 是否返回归一化系数
    :return:
    """
    waveform, sr = read_audio(audio_path)
    if len(waveform) < time_length * sr:
        # padding
        waveform = tf.concat([waveform, tf.zeros([time_length * sr - len(waveform), 1])], axis=0)
    else:
        # cut
        waveform = waveform[:time_length * sr]

    # emphasis audio_shape=(1, 16000)
    pre_emphasis = 0.97
    emphasis_waveform = tf.concat([waveform[0:1], waveform[1:] - pre_emphasis * waveform[:-1]], axis=0)

    # norm
    if if_norm:
        if norm_factor is None:
            # waveform = waveform_norm(tf.squeeze(waveform, axis=-1))
            waveform, norm_factor = waveform_norm(tf.transpose(emphasis_waveform, perm=[1, 0]))
            waveform = tf.squeeze(waveform, axis=0)
        else:
            waveform = waveform_norm(tf.transpose(emphasis_waveform, perm=[1, 0]), norm_factor=norm_factor)
            waveform = tf.squeeze(waveform, axis=0)

    if out_type == 'waveform' and return_norm:
        return waveform, norm_factor
    elif out_type == 'waveform' and not return_norm:
        return waveform

    if out_type == 'complex':
        return wave_complex(waveform)

    elif out_type == 'frames':
        # frame
        frame_length = 510
        frame_step = 255
        frames = tf.signal.frame(waveform, frame_length, frame_step, axis=0)
        return tf.squeeze(frames, axis=-1)
    elif out_type == 'magnitude':
        # waveform = tf.transpose(waveform, perm=[1, 0])
        # waveform = tf.expand_dims(waveform, axis=0)
        waveform = tf.squeeze(waveform, axis=-1)
        stft = tf.signal.stft(waveform, frame_length=510, frame_step=255, fft_length=510)
        # 计算幅度谱和相位谱
        magnitude = tf.abs(stft)
        phase = tf.math.angle(stft)
        if power_compress:
            magnitude = tf.pow(magnitude, 0.3)
        return magnitude, phase

def func_audios(audio_paths, if_norm=True, out_type='waveform', return_norm=False):
    if out_type == 'magnitude':
        mags = []
        phases = []
        for audio_path in audio_paths:
            mag, phase = func_audio(audio_path, out_type=out_type, return_norm=return_norm)
            mags.append(tf.expand_dims(mag, axis=0))
            phases.append(tf.expand_dims(phase, axis=0))
        return tf.concat([tf.concat(mags, axis=0), tf.concat(phases, axis=0)], axis=-1)

    if out_type == 'waveform':
        waves = []
        for audio_path in audio_paths:
            waves.append(tf.expand_dims(func_audio(audio_path, if_norm=if_norm, out_type=out_type, return_norm=return_norm), axis=0))
        waves = tf.concat(waves, axis=0)
        return waves
    elif out_type == 'complex':
        waves = []
        for audio_path in audio_paths:
            waves.append(tf.expand_dims(func_audio(audio_path, if_norm=if_norm, out_type='waveform'), axis=0))
        waves = tf.concat(waves, axis=0)
        complex_domain = wave_complex(waves)
        return complex_domain

def func_union_audios(union_audio_paths, if_norm, out_type):
    union_out = []
    for i in range(len(union_audio_paths)):
        clean_path_i, noisy_path_i = union_audio_paths[i]
        noisy_waveform_i, noisy_norm_factor = func_audio(noisy_path_i, if_norm=if_norm, out_type=out_type, return_norm=True)
        clean_waveform_i, clean_norm_factor = func_audio(clean_path_i, if_norm=if_norm, out_type=out_type, return_norm=True, norm_factor=noisy_norm_factor)

        if out_type == 'complex':
            union_i = tf.concat([clean_waveform_i, noisy_waveform_i], axis=-1)
        elif out_type == 'waveform':
            clean_waveform_i = tf.expand_dims(clean_waveform_i, axis=-1)
            noisy_waveform_i = tf.expand_dims(noisy_waveform_i, axis=-1)
            union_i = tf.concat([clean_waveform_i, noisy_waveform_i], axis=-1)
        union_out.append(tf.expand_dims(union_i, axis=0))
    union_out = tf.concat(union_out, axis=0)
    return union_out

# 指定源文件夹和目标文件夹的路径
noisy_pink = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/second_train/noisy_pink/"
noisy_babble = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/second_train/noisy_babble/"
noisy_hfchannel = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/second_train/noisy_hfchannel/"
noisy_factory1 = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/second_train/noisy_factory1/"
SNR = [-5, 0, 5, 10]
noisy_list = [noisy_pink, noisy_babble, noisy_hfchannel, noisy_factory1]
noisy_speeches = []
for noisy in noisy_list:
    for snr in SNR:
        noisy_speeches += tf.io.gfile.glob(noisy + 'noisy_' + str(snr) + '/*.wav')

clean_dir = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/second_train/clean_speech/"
clean_speech = tf.io.gfile.glob(clean_dir + '*.wav')
clean_speeches = []
for i in range(16):
    clean_speeches += clean_speech

def path_union(x_path, y_path):
    z_path = []
    for i in range(len(x_path)):
        z_path.append((x_path[i], y_path[i]))
    return z_path

#shuffle the data
train_X, test_X, train_y, test_y = train_test_split(noisy_speeches, clean_speeches, test_size=0.2, random_state=42, shuffle=True)

rate = 0.3
train_length = int(len(train_X) * rate)
test_length = int(len(test_X) * rate)
train_X = train_X[:train_length]
train_y = train_y[:train_length]
test_X = test_X[:test_length]
test_y = test_y[:test_length]
train_noisy_clean = path_union(train_X, train_y)
test_noisy_clean = path_union(test_X, test_y)

audio_train = func_union_audios(train_noisy_clean, if_norm=True, out_type='waveform')
audio_test = func_union_audios(test_noisy_clean, if_norm=True, out_type='waveform')

np_train_X, np_train_y = tf.split(audio_train, 2, axis=-1)
np_test_X, np_test_y = tf.split(audio_test, 2, axis=-1)
np_train_X = tf.squeeze(np_train_X, axis=-1)
np_train_y = tf.squeeze(np_train_y, axis=-1)
np_test_X = tf.squeeze(np_test_X, axis=-1)
np_test_y = tf.squeeze(np_test_y, axis=-1)

# Optimizers & LR schedulers
lr_batch = lr_batch = len(np_train_X) // Batch_size
first_gap = 3
second_gap = 5
learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [lr_batch * first_gap, lr_batch * second_gap], [learning_rate, learning_rate * 1e-1, learning_rate * 1e-2])
wd_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [lr_batch * first_gap, lr_batch * second_gap], [learning_rate, learning_rate * 1e-1, learning_rate * 1e-2])

G_optimizer = AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule, beta_1=0.5)
finetune_model.new_compile(gen_optimizer=G_optimizer,
                           speech_loss=loss_func,
                           noisy_loss=mse_pred)

# Load your data into NumPy arrays, e.g., x_train, y_train
# ...

# Create a TensorFlow Dataset from the NumPy arrays
train_dataset = tf.data.Dataset.from_tensor_slices((np_train_X, np_train_y))
# Shuffle, batch, and prefetch the dataset
BATCH_SIZE = Batch_size
train_dataset = train_dataset.shuffle(10).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create a TensorFlow Dataset from the NumPy arrays
test_dataset = tf.data.Dataset.from_tensor_slices((np_test_X, np_test_y))
# Batch and prefetch the test dataset
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Train the model using the prepared dataset
history = finetune_model.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=[cp_callback])
