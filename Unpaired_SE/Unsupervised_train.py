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


inputTime_size = 249
inputFreq_size = 256
epochs = 10
Batch_size = 2
end = 8000
start = 0


generator = tf_models.Gen_model(de_speech=tf_models.Generator_speech(kernel_size=4, strides=2, feature_timeaxis=124),
                                de_noise=tf_models.Gen_AIA(kernel_size=[3, 4, 4], strides=[1, 2, 2], filters=[128, 64, 32, 256],
                                                           gate_filter=32, gate_kernel_1=5, gate_kernel_2=1))

discriminator_noise = discriminator.Discriminator(ndf=64)
discriminator_noisy = discriminator.Discriminator(ndf=64)
discriminator_speech = discriminator.Discriminator(ndf=64)

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
    loss = 0
    for i, window_size in enumerate(window_sizes):
        stft_true = tf.signal.stft(y_true, frame_length=window_size, frame_step=window_size // 2,
                                   fft_length=window_size)
        stft_pred = tf.signal.stft(y_pred, frame_length=window_size, frame_step=window_size // 2,
                                   fft_length=window_size)
        mag_true = tf.abs(stft_true)
        mag_pred = tf.abs(stft_pred)
        # diff = tf.abs(stft_true - stft_pred)
        # diff_mag = tf.math.log(mag_true + 1e-6) - tf.math.log(mag_pred + 1e-6)
        # loss += weights[i] * tf.reduce_mean(diff)
        diff_mag = tf.keras.losses.MeanAbsoluteError()(y_true=mag_true, y_pred=mag_pred)
        loss += weights[i] * diff_mag 
    return loss + diff_sisdr

def generator_loss(label, fake_pred):
    return mse(label, fake_pred)

def mae_pred(y_true, y_pred):
    return K.abs(K.mean(y_true - y_pred))

def mse_pred(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

def loss_func(y_true, y_pred, alpha=0.8, beta=0.1, gamma=0.1, input_mean=None, input_std=None):
    # min_length = tf.minimum(tf.shape(y_true)[-1], tf.shape(y_pred)[-1])
    # y_true = y_true[:, :min_length]
    # y_pred = y_pred[:, :min_length]

    loss_sisdr = multi_resolution_stft_loss(y_true=y_true, y_pred=y_pred,
                                            window_sizes=[256, 512, 1024, 2048], weights=[1, 0.5, 0.25, 0.125],
                                            if_frames=False, mean=input_mean, std=input_std)

    loss_mae = mae_pred(y_true=y_true, y_pred=y_pred)
    loss_mse = mse_pred(y_true=y_true, y_pred=y_pred)
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
    def __init__(self, generator, discriminator_speech, discriminator_noisy, discriminator_noise, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator_speech = discriminator_speech
        self.discriminator_noisy = discriminator_noisy
        self.discriminator_noise = discriminator_noise

    def new_compile(self, gen_optimizer,
                    disc_speech_optimizer,
                    disc_noisy_optimizer,
                    disc_noise_optimizer,
                    generator_loss,
                    Generator_loss,
                    Identity_loss,
                    reconstruction_loss):
        super().compile()

        # Optimizers
        self.gen_optimizer = gen_optimizer
        self.disc_speech_optimizer = disc_speech_optimizer
        self.disc_noisy_optimizer = disc_noisy_optimizer
        self.disc_noise_optimizer = disc_noise_optimizer

        # Losses
        self.generator_loss = generator_loss
        self.Generator_loss = Generator_loss
        self.Identity_loss = Identity_loss
        self.reconstruction_loss = reconstruction_loss

        # Trackers
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
        self.disc_speech_tracker = keras.metrics.Mean(name="disc_loss")
        self.disc_noisy_tracker = keras.metrics.Mean(name="disc_loss")
        self.disc_noise_tracker = keras.metrics.Mean(name="disc_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.id_loss_tracker = keras.metrics.Mean(name="id_loss")
        self.confidence_loss_tracker = keras.metrics.Mean(name="confidence_loss")

    # def build(self, input_shape):
    #       self.inp = keras.Input(input_shape)

    def train_step(self, batch, training=True):
        x_batch_train, y_batch_train = batch
        split_input = tf.split(x_batch_train, num_or_size_splits=2, axis=-1)
        clean_speech, noisy_speech = tf.squeeze(split_input[0], axis=-1), tf.squeeze(split_input[1], axis=-1)
        noise = y_batch_train
        complex_clean_speech = wave_complex(clean_speech)
        complex_noisy_speech = wave_complex(noisy_speech)
        # complex_noise = wave_complex(noise)

        with tf.GradientTape(persistent=True) as disc_tape, tf.GradientTape() as gen_tape:
            out_gen_noisy = self.generator(complex_noisy_speech)
            out_gen_clean = self.generator(complex_clean_speech)
            gen_noisy_speech = out_gen_noisy[0]
            gen_noisy_noise = out_gen_noisy[1]
            gen_clean_speech = out_gen_clean[0]
            audio_gen_clean_speech = complex_wave(gen_clean_speech, frame_length=510, frame_step=255, fft_length=510)
            audio_gen_noisy_speech = complex_wave(gen_noisy_speech, frame_length=510, frame_step=255, fft_length=510)
            audio_gen_noisy_noise = complex_wave(gen_noisy_noise, frame_length=510, frame_step=255, fft_length=510)
            audio_gen_noisy_noisy = audio_gen_noisy_noise + audio_gen_noisy_speech
            # gen_noisy_noisy = wave_complex(audio_gen_noisy_noisy, frame_length=510, frame_step=255, fft_length=510)
            mag_gen_noisy_noisy = wave_mag(audio_gen_noisy_noisy, frame_length=510, frame_step=255, fft_length=510)
            mag_gen_noisy_speech = complex_mag(gen_noisy_speech, frame_length=510, frame_step=255, fft_length=510)
            mag_gen_noisy_noise = complex_mag(gen_noisy_noise, frame_length=510, frame_step=255, fft_length=510)
            mag_gen_clean_speech = complex_mag(gen_clean_speech, frame_length=510, frame_step=255, fft_length=510)
            mag_clean_speech = wave_mag(clean_speech, frame_length=510, frame_step=255, fft_length=510)
            mag_noisy_speech = wave_mag(noisy_speech, frame_length=510, frame_step=255, fft_length=510)
            mag_noise = wave_mag(noise, frame_length=510, frame_step=255, fft_length=510)
            # 计算pesq
            # pesq_score = discriminator.batch_pesq(clean=clean_speech[:31875], noisy=audio_gen_clean_speech)

            # Generating the features using the discriminator
            fake_Speech_label = self.discriminator_speech(mag_clean_speech, mag_gen_clean_speech) #out shape is (batch_size, 1)
            fake_speech_label = self.discriminator_speech(mag_clean_speech, mag_gen_noisy_speech)
            fake_noisy_label = self.discriminator_noisy(mag_noisy_speech, mag_gen_noisy_noisy)
            fake_noise_label = self.discriminator_noise(mag_noise, mag_gen_noisy_noise)

            real_speech_label = self.discriminator_speech(mag_clean_speech, mag_clean_speech)
            real_noisy_label = self.discriminator_noisy(mag_noisy_speech, mag_noisy_speech)
            real_noise_label = self.discriminator_noise(mag_noise, mag_noise)

            label_real_noise = tf.ones((Batch_size, 1))
            label_real_speech = tf.ones((Batch_size, 1))
            label_real_noisy = tf.ones((Batch_size, 1))

            label_fake_noise = tf.zeros((Batch_size, 1))
            label_fake_speech = tf.zeros((Batch_size, 1))
            label_fake_noisy = tf.zeros((Batch_size, 1))

            # Calculating the discriminator losses

            Dloss_noisy_fake = self.generator_loss(y_true=label_fake_noisy, y_pred=fake_noisy_label)
            Dloss_noise_fake = self.generator_loss(y_true=label_fake_noise, y_pred=fake_noise_label)
            Dloss_speech_fake = self.generator_loss(y_true=label_fake_speech, y_pred=fake_speech_label)
            Dloss_Speech_fake = self.generator_loss(y_true=label_fake_speech, y_pred=fake_Speech_label)
            Dloss_noisy_real = self.generator_loss(y_true=label_real_noisy, y_pred=real_noisy_label)
            Dloss_noise_real = self.generator_loss(y_true=label_real_noise, y_pred=real_noise_label)
            Dloss_speech_real = self.generator_loss(y_true=label_real_speech, y_pred=real_speech_label)

            disc_speech_loss = Dloss_speech_real + Dloss_Speech_fake + Dloss_speech_fake
            disc_noise_loss = Dloss_noise_real + Dloss_noise_fake
            disc_noisy_loss = Dloss_noisy_real + Dloss_noisy_fake
            disc_loss = disc_speech_loss + disc_noise_loss + disc_noisy_loss

            # Calculating and applying the gradients for discriminator
            grads_disc_speech = disc_tape.gradient(disc_speech_loss, self.discriminator_speech.trainable_weights)
            grads_disc_noise = disc_tape.gradient(disc_noise_loss, self.discriminator_noise.trainable_weights)
            grads_disc_noisy = disc_tape.gradient(disc_noisy_loss, self.discriminator_noisy.trainable_weights)
            self.disc_speech_optimizer.apply_gradients(zip(grads_disc_speech, self.discriminator_speech.trainable_weights))
            self.disc_noisy_optimizer.apply_gradients(zip(grads_disc_noisy, self.discriminator_noisy.trainable_weights))
            self.disc_noise_optimizer.apply_gradients(zip(grads_disc_noise, self.discriminator_noise.trainable_weights))

            self.disc_loss_tracker.update_state(disc_loss)
            self.disc_speech_tracker.update_state(disc_speech_loss)
            self.disc_noise_tracker.update_state(disc_noise_loss)
            self.disc_noisy_tracker.update_state(disc_noisy_loss)

            # Calculating the generator losses
            y_labels_1 = tf.ones((Batch_size, 1))
            y_labels_2 = tf.ones((Batch_size, 1))
            y_labels_3 = tf.ones((Batch_size, 1))

            out_gen = self.generator(complex_noisy_speech)
            # the output of the generator is a tuple of two tensors: the generated speech and the generated noise,
            # they are also the complex domain
            speech_gen = out_gen[0]
            noise_gen = out_gen[1]
            audio_speech_gen = complex_wave(speech_gen, frame_length=510, frame_step=255, fft_length=510)
            audio_noise_gen = complex_wave(noise_gen, frame_length=510, frame_step=255, fft_length=510)
            audio_noisy_gen = audio_speech_gen + audio_noise_gen
            mag_speech_gen = wave_mag(audio_speech_gen, frame_length=510, frame_step=255, fft_length=510)
            mag_noise_gen = wave_mag(audio_noise_gen, frame_length=510, frame_step=255, fft_length=510)
            mag_noisy_gen = wave_mag(audio_noisy_gen, frame_length=510, frame_step=255, fft_length=510)

            # noisy_gen = wave_complex(audio_noisy_gen, frame_length=510, frame_step=255, fft_length=510)

            confidence_noise = self.discriminator_noise(mag_noise, mag_noise_gen, training=False)
            confidence_noisy = self.discriminator_noisy(mag_noisy_speech, mag_noisy_gen, training=False)
            confidence_speech = self.discriminator_speech(mag_clean_speech, mag_speech_gen, training=False)

            # generator_loss=generator_loss
            confidence_loss_speech = self.generator_loss(y_true=y_labels_1, y_pred=confidence_speech)
            confidence_loss_noisy = self.generator_loss(y_true=y_labels_2, y_pred=confidence_noisy)
            confidence_loss_noise = self.generator_loss(y_true=y_labels_3, y_pred=confidence_noise)
            Confidence_loss = confidence_loss_speech + confidence_loss_noisy + confidence_loss_noise

            # Generator_loss=loss_func  return alpha * loss_sisdr + beta * loss_mae + gamma * loss_mse
            # min_length_speech = min(int(clean_speech.shape[-1]), int(audio_speech_gen.shape[-1]))
            # min_length_noisy = min(int(noisy_speech.shape[-1]), int(audio_noisy_gen.shape[-1]))
            # min_length_noise = min(int(noise.shape[-1]), int(audio_noise_gen.shape[-1]))

            min_length_speech = 31875
            min_length_noisy = 31875
            min_length_noise = 31875
            confidence_loss = self.Generator_loss(y_true=clean_speech[:, :min_length_speech], y_pred=audio_speech_gen[:, :min_length_speech], alpha=0.8, beta=0.1,
                                                  gamma=0.1, input_mean=None, input_std=None) + \
                              self.Generator_loss(y_true=noisy_speech[:, :min_length_noisy], y_pred=audio_noisy_gen[:, :min_length_noisy], alpha=0.4, beta=0.3, gamma=0.3,
                                                  input_mean=None, input_std=None) + \
                              self.Generator_loss(y_true=noise[:, :min_length_noise], y_pred=audio_noise_gen[:, :min_length_noise], alpha=0, beta=0.5, gamma=0.5,
                                                  input_mean=None, input_std=None) + Confidence_loss

            # Identity_loss=Identity_loss return wave_loss * 0.99 + mel_loss * 0.01
            recon_loss = tf.reduce_mean(self.reconstruction_loss(y_true=noisy_speech[:, :min_length_noisy], y_pred=audio_noisy_gen[:, :min_length_noisy], if_cosine=True))
            # id_loss = tf.reduce_mean(
            #     self.Identity_loss(y_true=clean_speech[:, :min_length_speech], y_pred=audio_speech_gen[:, :min_length_speech], input_mean=None, input_std=None,
            #                        if_cosine=True))
            id_loss = self.Identity_loss(y_true=clean_speech[:, :min_length_speech], y_pred=audio_speech_gen[:, :min_length_speech])
            gen_loss = confidence_loss + 0.2 * recon_loss + id_loss
            # Calculating and applying the gradients for generator
            grads_gen = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
            self.gen_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_weights))

            self.gen_loss_tracker.update_state(gen_loss)
            self.recon_loss_tracker.update_state(recon_loss)
            self.id_loss_tracker.update_state(id_loss)
            self.confidence_loss_tracker.update_state(confidence_loss)

            return {
                "gen_loss": self.gen_loss_tracker.result(),
                "confidence_loss": self.confidence_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "id_loss": self.id_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result(),
                "disc_speech_loss": self.disc_speech_tracker.result(),
                "disc_noisy_loss": self.disc_noisy_tracker.result(),
                "disc_noise_loss": self.disc_noise_tracker.result()
            }
        # implement the call method

    def call(self, inputs, *args, **kwargs):
        # pass
        split_input = tf.split(inputs, num_or_size_splits=2, axis=-1)
        clean_speech, noisy_speech = tf.squeeze(split_input[0], axis=-1), tf.squeeze(split_input[1], axis=-1)
        complex_clean_speech = wave_complex(clean_speech)
        complex_noisy_speech = wave_complex(noisy_speech)
        return self.generator(complex_noisy_speech)

#不训练finetune模型的时候，需要把这个注释掉
this_model = Jianqiao_GAN(generator=generator,
                          discriminator_speech=discriminator_speech,
                          discriminator_noisy=discriminator_noisy,
                          discriminator_noise=discriminator_noise)


# # 指定源文件夹和目标文件夹的路径
# noisy_pink = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/first_train/noisy_pink/"
# noisy_babble = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/first_train/noisy_babble/"
# noisy_hfchannel = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/first_train/noisy_hfchannel/"
# noisy_factory1 = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/first_train/noisy_factory1/"
# noise_folder = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/first_train/noise/"


# noisy_speeches = tf.io.gfile.glob(noisy_pink + '*.wav') + tf.io.gfile.glob(noisy_babble + '*.wav') + \
#                     tf.io.gfile.glob(noisy_hfchannel + '*.wav') + tf.io.gfile.glob(noisy_factory1 + '*.wav')

# clean_dir = "/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/first_train/clean_split_2s/"
# clean_speeches = tf.io.gfile.glob(clean_dir + '*.wav')[: len(noisy_speeches)]

# # 遍历源文件夹中的所有文件
# noises = []
# for file_name in noisy_speeches:
#     if file_name.endswith(".wav"):
#         # noises.append(os.path.join(noise_folder, file_name.split("_")[0] + ".wav"))
#         noises.append(os.path.join(noise_folder, file_name.split("/")[-1].split("_")[0] + ".wav"))

# # clean_speeches = []
# # for i in range(4):
# #     clean_speeches += clean_paths

# def path_union(x_path, y_path):
#     z_path = []
#     for i in range(len(x_path)):
#         z_path.append((x_path[i], y_path[i]))
#     return z_path
# clean_noisy_union = path_union(clean_speeches, noisy_speeches)
# #shuffle the data
# train_X, test_X, train_y, test_y = train_test_split(clean_noisy_union, noises, test_size=0.2, random_state=42, shuffle=True)

# div_rate =0.3
# train_length = int(len(train_X) * div_rate)
# train_X = train_X[:train_length]
# train_y = train_y[:train_length]
# test_length = int(len(test_X) * div_rate)
# test_X = test_X[:test_length]
# test_y = test_y[:test_length]

# # Optimizers & LR schedulers
# first_gap = 3
# second_gap = 5
# lr_batch = len(train_X) // Batch_size
# learning_rate = 1e-4
# genlr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#     [lr_batch * first_gap, lr_batch * second_gap], [learning_rate, learning_rate * 1e-1, learning_rate * 1e-2])
# genwd_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#     [lr_batch * first_gap, lr_batch * second_gap], [learning_rate, learning_rate * 1e-1, learning_rate * 1e-2])
# # genbiaslr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
# #     [lr_batch * first_gap, lr_batch * second_gap], [2 * learning_rate * 1e2, 4 * learning_rate * 1e1, 8 * learning_rate * 1e1])
# dislr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#     [lr_batch * first_gap, lr_batch * second_gap], [learning_rate, learning_rate * 1e-1, learning_rate * 1e-2])
# diswd_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#     [lr_batch * first_gap, lr_batch * second_gap], [learning_rate, learning_rate * 1e-1, learning_rate * 1e-2])
# # disbiaslr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
# #     [lr_batch * first_gap, lr_batch * second_gap], [2 * learning_rate * 1e2, 4 * learning_rate * 1e1, 8 * learning_rate * 1e1])


# G_optimizer = AdamW(learning_rate=genlr_schedule, weight_decay=genwd_schedule, beta_1=0.5)
# D_speech_optimizer = AdamW(learning_rate=dislr_schedule, weight_decay=diswd_schedule, beta_1=0.5)
# D_noise_optimizer = AdamW(learning_rate=dislr_schedule, weight_decay=diswd_schedule, beta_1=0.5)
# D_noisy_optimizer = AdamW(learning_rate=dislr_schedule, weight_decay=diswd_schedule, beta_1=0.5)

# this_model = Jianqiao_GAN(generator=generator,
#                           discriminator_speech=discriminator_speech,
#                           discriminator_noisy=discriminator_noisy,
#                           discriminator_noise=discriminator_noise)

# this_model.new_compile(gen_optimizer=G_optimizer,
#                        disc_speech_optimizer=D_speech_optimizer,
#                        disc_noisy_optimizer=D_noisy_optimizer,
#                        disc_noise_optimizer=D_noise_optimizer,
#                        generator_loss=mse_pred,
#                        Generator_loss=loss_func,
#                        Identity_loss=mse_pred,
#                        reconstruction_loss=reconstruction_loss)


# checkpoint_save_path = "./pre_train_param_1/model.ckpt"
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the pre_train model-----------------')
#     this_model.load_weights(checkpoint_save_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                 save_weights_only=True,
#                                                 save_best_only=True,
#                                                 monitor='val_loss',
#                                                 mode='auto',
#                                                 verbose=1)


# def read_audio(audio_path):
#     audio_binary = tf.io.read_file(audio_path)
#     waveform, sample_rate = tf.audio.decode_wav(audio_binary)
#     return waveform, sample_rate

# def waveform_norm(waveform, norm_factor=None):
#     #waveform shape: [time, sr * time]
#     if norm_factor is None:
#         c = tf.sqrt(tf.cast(tf.shape(waveform)[-1], tf.float32) / tf.reduce_sum(waveform ** 2.0, axis=-1))
#         waveform = tf.transpose(waveform, perm=[1, 0])
#         waveform = tf.transpose(waveform * c, perm=[1, 0])
#         return waveform, c
#     else:
#         waveform = tf.transpose(waveform, perm=[1, 0])
#         waveform = tf.transpose(waveform * norm_factor, perm=[1, 0])
#         return waveform

# def func_audio(audio_path, time_length=2, if_norm=False,
#                out_type='waveform', power_compress=True,
#                norm_factor=None, return_norm=True):
#     """
#     :param audio_path: 读取的音频路径
#     :param time_length: 音频长度，单位为秒，如果音频长度小于time_length，则进行padding，否则进行截断
#     :param if_norm: 是否进行归一化
#     :param out_type: 输出的特征类型，可选为'waveform'、'spectrogram'、'mel_spectrogram'、'mfcc'
#     :param power_compress: 是否进行power_compress
#     :param norm_factor: 传入的归一化系数
#     :param return_norm: 是否返回归一化系数
#     :return:
#     """
#     waveform, sr = read_audio(audio_path)
#     if len(waveform) < time_length * sr:
#         # padding
#         waveform = tf.concat([waveform, tf.zeros([time_length * sr - len(waveform), 1])], axis=0)
#     else:
#         # cut
#         waveform = waveform[:time_length * sr]

#     # emphasis audio_shape=(1, 16000)
#     pre_emphasis = 0.97
#     emphasis_waveform = tf.concat([waveform[0:1], waveform[1:] - pre_emphasis * waveform[:-1]], axis=0)

#     # norm
#     if if_norm:
#         if norm_factor is None:
#             # waveform = waveform_norm(tf.squeeze(waveform, axis=-1))
#             waveform, norm_factor = waveform_norm(tf.transpose(emphasis_waveform, perm=[1, 0]))
#             waveform = tf.squeeze(waveform, axis=0)
#         else:
#             waveform = waveform_norm(tf.transpose(emphasis_waveform, perm=[1, 0]), norm_factor=norm_factor)
#             waveform = tf.squeeze(waveform, axis=0)

#     if out_type == 'waveform' and return_norm:
#         return waveform, norm_factor
#     elif out_type == 'waveform' and not return_norm:
#         return waveform

#     if out_type == 'complex':
#         return wave_complex(waveform)

#     elif out_type == 'frames':
#         # frame
#         frame_length = 510
#         frame_step = 255
#         frames = tf.signal.frame(waveform, frame_length, frame_step, axis=0)
#         return tf.squeeze(frames, axis=-1)
#     elif out_type == 'magnitude':
#         # waveform = tf.transpose(waveform, perm=[1, 0])
#         # waveform = tf.expand_dims(waveform, axis=0)
#         waveform = tf.squeeze(waveform, axis=-1)
#         stft = tf.signal.stft(waveform, frame_length=510, frame_step=255, fft_length=510)
#         # 计算幅度谱和相位谱
#         magnitude = tf.abs(stft)
#         phase = tf.math.angle(stft)
#         if power_compress:
#             magnitude = tf.pow(magnitude, 0.3)
#         return magnitude, phase

# def func_audios(audio_paths, if_norm=True, out_type='waveform', return_norm=False):
#     if out_type == 'magnitude':
#         mags = []
#         phases = []
#         for audio_path in audio_paths:
#             mag, phase = func_audio(audio_path, out_type=out_type, return_norm=return_norm)
#             mags.append(tf.expand_dims(mag, axis=0))
#             phases.append(tf.expand_dims(phase, axis=0))
#         return tf.concat([tf.concat(mags, axis=0), tf.concat(phases, axis=0)], axis=-1)

#     if out_type == 'waveform':
#         waves = []
#         for audio_path in audio_paths:
#             waves.append(tf.expand_dims(func_audio(audio_path, if_norm=if_norm, out_type=out_type, return_norm=return_norm), axis=0))
#         waves = tf.concat(waves, axis=0)
#         return waves
#     elif out_type == 'complex':
#         waves = []
#         for audio_path in audio_paths:
#             waves.append(tf.expand_dims(func_audio(audio_path, if_norm=if_norm, out_type='waveform'), axis=0))
#         waves = tf.concat(waves, axis=0)
#         complex_domain = wave_complex(waves)
#         return complex_domain

# def func_union_audios(union_audio_paths, if_norm, out_type):
#     union_out = []
#     for i in range(len(union_audio_paths)):
#         clean_path_i, noisy_path_i = union_audio_paths[i]
#         noisy_waveform_i, noisy_norm_factor = func_audio(noisy_path_i, if_norm=if_norm, out_type=out_type, return_norm=True)
#         clean_waveform_i, clean_norm_factor = func_audio(clean_path_i, if_norm=if_norm, out_type=out_type, return_norm=True, norm_factor=noisy_norm_factor)

#         if out_type == 'complex':
#             union_i = tf.concat([clean_waveform_i, noisy_waveform_i], axis=-1)
#         elif out_type == 'waveform':
#             clean_waveform_i = tf.expand_dims(clean_waveform_i, axis=-1)
#             noisy_waveform_i = tf.expand_dims(noisy_waveform_i, axis=-1)
#             union_i = tf.concat([clean_waveform_i, noisy_waveform_i], axis=-1)
#         union_out.append(tf.expand_dims(union_i, axis=0))
#     union_out = tf.concat(union_out, axis=0)
#     return union_out

# # Load your data into NumPy arrays, e.g., x_train, y_train
# np_train_X = func_union_audios(train_X, if_norm=True, out_type='waveform')
# np_test_X = func_union_audios(test_X, if_norm=True, out_type='waveform')
# np_train_y = func_audios(train_y, if_norm=True, out_type='waveform')
# np_test_y = func_audios(test_y, if_norm=True, out_type='waveform')

# # Create a TensorFlow Dataset from the NumPy arrays
# train_dataset = tf.data.Dataset.from_tensor_slices((np_train_X, np_train_y))
# # Shuffle, batch, and prefetch the dataset
# BATCH_SIZE = Batch_size
# train_dataset = train_dataset.shuffle(10).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# # Create a TensorFlow Dataset from the NumPy arrays
# test_dataset = tf.data.Dataset.from_tensor_slices((np_test_X, np_test_y))
# # Batch and prefetch the test dataset
# test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# # Train the model using the prepared dataset
# history = this_model.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=[cp_callback])
