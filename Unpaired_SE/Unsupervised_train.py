#!/mainfs/home/jc18n17/anaconda3/envs/TF2.6/bin/python3.6
"""
# 这部分的speech encoder是用的ECAPA-TDNN，用的是tf_models.py里面的ECAPA_TDNN（其中input feature是waveform，然后转变为mel- spectrogram），
# 紧接着利用attention机制将ECAPA_TDNN的输出（mel-spectrogram）和original（waveform）融合成了一个256维的向量
# 这部分的noise decoder是用的Gen_aia，用的是tf_models.py里面的Gen_aia
# 这部分的speech decoder是用的Generator_speech，用的是tf_models.py里面的glu_unit的model_unit
# 然后把这三个decoder和speech encoder组合成了一个generator，用的是tf_models.py里面的Gen_model
"""
import tensorflow as tf
from keras import Model
import keras.backend as K
import numpy as np
from tensorflow import keras
from tensorflow_addons.optimizers import AdamW
import os
import tf_models
import evaluation
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import glu_unit

# 启用混合精度训练
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

# define generators and corresponding discriminators
inputTime_size = 124
inputFreq_size = 256
epochs = 10
batch_size = 2
end = 40000
start = 0


encoder_speech = tf_models.ECAPA_TDNN(C=256, scale=4)
decoder_speech = tf_models.Generator_speech(kernel_size=7, strides=(1, 2), feature_timeaxis=inputTime_size)
# decoder_speech = glu_unit.model_unit
decoder_noise = tf_models.Gen_AIA(kernel_size=[(1,3),(3,5),(3,5)], strides=[(1,1),(1,2),(1,2)],filters=[16,16,64,1],
                gate_filter=64, gate_kernel_1=5, gate_kernel_2=1)

generator = tf_models.Gen_model(en_speech=encoder_speech,
                  de_speech=decoder_speech,
                  de_noise=decoder_noise)

discriminator_noise = tf_models.Create_discriminator((inputTime_size, inputFreq_size), if_mel= False)
discriminator_noisy = tf_models.Create_discriminator((inputTime_size, inputFreq_size), if_mel= False)
discriminator_speech = tf_models.Create_discriminator((inputTime_size, inputFreq_size), if_mel= True)


# define their own losses
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
cosine_loss = tf.keras.losses.CosineSimilarity(axis=-1)

## frames to wave
def inverse_framing(sig, frame_shift=128):
    return tf.signal.overlap_and_add(signal=sig, frame_step=frame_shift, name=None)

def func_mel(wave, sample_rate = 16000):
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

def mag_wave(mag, phase, win_len, step_len):
    stft = tf.complex(mag * tf.cos(phase), mag * tf.sin(phase))
    audio = tf.signal.inverse_stft(stft, frame_length=win_len, frame_step=step_len, fft_length=win_len)
    return audio

def multi_resolution_stft_loss(y_true, y_pred, window_sizes=[256, 512, 1024, 2048], weights=[1, 0.5, 0.25, 0.125],
                               if_frames=True, mean=None, std=None):
    """
    enhanced_speech_info = tf.concat([speech_gen, noisy_phase], axis=-1)
    clean_speech_info = tf.concat([clean_speech, clean_phase], axis=-1)
    enhanced_noisy_info = tf.concat([noisy_gen, noisy_phase], axis=-1)
    noisy_speech_info = tf.concat([noisy_speech, noisy_phase], axis=-1)
    enhanced_noise_info = tf.concat([noise_gen, noisy_phase], axis=-1)
    noise_info = tf.concat([noise, noise_phase], axis=-1)
    y_true = original_info
    y_pred = enhanced_info
    """
    original_mag, original_phase = tf.split(y_true, num_or_size_splits=2, axis=-1)
    enhanced_mag, enhanced_phase = tf.split(y_pred, num_or_size_splits=2, axis=-1)

    y_true = mag_wave(mag=original_mag, phase=original_phase, win_len=510, step_len=255)
    y_pred = mag_wave(mag=enhanced_mag, phase=enhanced_phase, win_len=510, step_len=255)

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
        diff_sisdr = evaluation.si_sdr(y_true=mag_true, y_pred=mag_pred)
        loss += weights[i] * tf.reduce_mean(diff_mag) + weights[i] * tf.reduce_mean(diff_sisdr)
    return loss

def generator_loss(label, fake_pred):
    gen_loss = []
    for i in range(len(fake_pred)):
        gen_loss.append(mse(label[i], fake_pred[i]))
    return tf.reduce_mean(gen_loss)

def gen_union_loss(labels, fake_preds):
    temp = []
    for i in range(len(fake_preds)):
        temp.append(generator_loss(fake_pred=fake_preds[i], label=labels[i]))
    return tf.reduce_mean(temp)

def gen_speech_loss(labels, fake_preds):
    return generator_loss(fake_pred=fake_preds[1], label=labels)

def mae_pred(y_true, y_pred):
    return K.abs(K.mean(y_true - y_pred))

def mse_pred(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

def loss_func(y_true, y_pred, alpha=0.8, beta=0.1, gamma=0.1, input_mean=None, input_std=None):
    loss_sisdr = multi_resolution_stft_loss(y_true=y_true, y_pred=y_pred,
                                            window_sizes=[256, 512, 1024, 2048], weights=[1, 0.5, 0.25, 0.125],
                                            if_frames=True, mean=input_mean, std=input_std)
    original_mag, original_phase = tf.split(y_true, num_or_size_splits=2, axis=-1)
    enhanced_mag, enhanced_phase = tf.split(y_pred, num_or_size_splits=2, axis=-1)

    loss_mae = mae_pred(y_true=original_mag, y_pred=enhanced_mag)
    loss_mse = mse_pred(y_true=original_mag, y_pred=enhanced_mag)
    return alpha * loss_sisdr + beta * loss_mae + gamma * loss_mse


def reconstruction_loss(y_true, y_pred, if_cosine=False):
    original_mag, original_phase = tf.split(y_true, num_or_size_splits=2, axis=-1)
    enhanced_mag, enhanced_phase = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    true_wave = mag_wave(mag=original_mag, phase=original_phase, win_len=510, step_len=255)
    pred_wave = mag_wave(mag=enhanced_mag, phase=enhanced_phase, win_len=510, step_len=255)
    if if_cosine:
        loss_cosine = cosine_loss(y_true=original_mag, y_pred=enhanced_mag)
        return mae_pred(y_pred=enhanced_mag, y_true=original_mag) + mse_pred(y_pred=pred_wave, y_true=true_wave) + loss_cosine
    else:
        return mae_pred(y_pred=enhanced_mag, y_true=original_mag) + mse_pred(y_pred=pred_wave, y_true=true_wave)

def Identity_loss(y_true, y_pred, input_mean=None, input_std=None, if_cosine=False):
    # if the input feature is raw waveform
    # wave_loss & mel_spec loss
    original_mag, original_phase = tf.split(y_true, num_or_size_splits=2, axis=-1)
    enhanced_mag, enhanced_phase = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    pred_mag = enhanced_mag
    true_mag = original_mag
    mag_loss = mse_pred(y_pred=pred_mag, y_true=true_mag)

    if input_mean is not None and input_std is not None:
        pred_outcome = pred_mag * input_std + input_mean
        true_outcome = true_mag * input_std + input_mean

    true_wave = mag_wave(mag=original_mag, phase=original_phase, win_len=510, step_len=255)
    pred_wave = mag_wave(mag=enhanced_mag, phase=enhanced_phase, win_len=510, step_len=255)

    pred_mel = func_mel(pred_wave)
    true_mel = func_mel(true_wave)

    mel_loss = 0.01 * mse_pred(y_true=true_mel, y_pred=pred_mel)

    if if_cosine:  # 加入余弦相似度来判定输出的语音跟目标的clean speech的相似度
        loss_cosine = cosine_loss(y_pred=pred_mel, y_true=true_mel)
        mel_loss = mel_loss + loss_cosine

    return mag_loss * 0.99 + mel_loss

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

    def train_step(self, batch):
        # X_train is noisy speech
        # y_train is clean speech
        # z_train is noise
        # they are all unpaired
        x_batch_train, y_batch_train = batch
        split_input = tf.split(x_batch_train, num_or_size_splits=4, axis=-1)
        clean_speech, clean_phase, noisy_speech, noisy_phase = split_input
        noise, noise_phase = tf.split(y_batch_train, num_or_size_splits=2, axis=-1)

        with tf.GradientTape(persistent=True) as disc_tape, tf.GradientTape() as gen_tape:
            out_gen_noisy = self.generator(noisy_speech)
            out_gen_clean = self.generator(clean_speech)
            gen_noisy_speech = out_gen_noisy[0]
            gen_noisy_noise = out_gen_noisy[1]
            gen_clean_speech = out_gen_clean[0]
            gen_noisy_noisy = gen_noisy_noise + gen_noisy_speech

            # Generating the features using the discriminator
            fake_Speech_label = self.discriminator_speech(gen_clean_speech)
            fake_speech_label = self.discriminator_speech(gen_noisy_speech)
            fake_noisy_label = self.discriminator_noisy(gen_noisy_noisy)
            fake_noise_label = self.discriminator_noise(gen_noisy_noise)

            real_speech_label = self.discriminator_speech(clean_speech)
            real_noisy_label = self.discriminator_noisy(noisy_speech)
            real_noise_label = self.discriminator_noise(noise)

            factor = 1#在time_axis=249的时候，factor=2，time_axis=124的时候，factor=1
            label_real_noise = [tf.ones((batch_size, 8 * factor, 1)), tf.ones((batch_size, 4* factor, 1)),
                                tf.ones((batch_size, 2* factor, 1))]
            label_real_speech = [tf.ones((batch_size, 8* factor, 1)), tf.ones((batch_size, 4* factor, 1)),
                                 tf.ones((batch_size, 2* factor, 1)),
                                 tf.ones((batch_size, 8* factor, 1)), tf.ones((batch_size, 4* factor, 1)),
                                 tf.ones((batch_size, 2* factor, 1))]
            label_real_noisy = [tf.ones((batch_size, 8* factor, 1)), tf.ones((batch_size, 4* factor, 1)),
                                tf.ones((batch_size, 2* factor, 1))]

            label_fake_noise = [tf.zeros((batch_size, 8* factor, 1)), tf.zeros((batch_size, 4* factor, 1)),
                                tf.zeros((batch_size, 2* factor, 1))]
            label_fake_speech = [tf.zeros((batch_size, 8* factor, 1)), tf.zeros((batch_size, 4* factor, 1)),
                                 tf.zeros((batch_size, 2* factor, 1)),
                                 tf.zeros((batch_size, 8* factor, 1)), tf.zeros((batch_size, 4* factor, 1)),
                                 tf.zeros((batch_size, 2* factor, 1))]
            label_fake_noisy = [tf.zeros((batch_size, 8* factor, 1)), tf.zeros((batch_size, 4* factor, 1)),
                                tf.zeros((batch_size, 2* factor, 1))]

            # Calculating the discriminator losses
            Dloss_noisy_fake = self.generator_loss(fake_pred=fake_noisy_label, label=label_fake_noisy)
            Dloss_noise_fake = self.generator_loss(fake_pred=fake_noise_label, label=label_fake_noise)
            Dloss_speech_fake = self.generator_loss(fake_pred=fake_speech_label, label=label_fake_speech)
            Dloss_Speech_fake = self.generator_loss(fake_pred=fake_Speech_label, label=label_fake_speech)
            Dloss_noisy_real = self.generator_loss(fake_pred=real_noisy_label, label=label_real_noisy)
            Dloss_noise_real = self.generator_loss(fake_pred=real_noise_label, label=label_real_noise)
            Dloss_speech_real = self.generator_loss(fake_pred=real_speech_label, label=label_real_speech)

            disc_speech_loss = tf.reduce_mean(Dloss_speech_real + Dloss_Speech_fake + Dloss_speech_fake)
            disc_noise_loss = tf.reduce_mean(Dloss_noise_real + Dloss_noise_fake)
            disc_noisy_loss = tf.reduce_mean(Dloss_noisy_real + Dloss_noisy_fake)
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
            y_labels_1 = [tf.ones((batch_size, 8* factor, 1)), tf.ones((batch_size, 4* factor, 1)),
                          tf.ones((batch_size, 2* factor, 1)),
                          tf.ones((batch_size, 8* factor, 1)), tf.ones((batch_size, 4* factor, 1)),
                          tf.ones((batch_size, 2* factor, 1))]
            y_labels_2 = [tf.ones((batch_size, 8* factor, 1)), tf.ones((batch_size, 4* factor, 1)),
                          tf.ones((batch_size, 2* factor, 1))]
            y_labels_3 = [tf.ones((batch_size, 8* factor, 1)), tf.ones((batch_size, 4* factor, 1)),
                          tf.ones((batch_size, 2* factor, 1))]

            out_gen = self.generator(noisy_speech)
            speech_gen = out_gen[0]
            noise_gen = out_gen[1]
            noisy_gen = speech_gen + noise_gen
            
            confidence_noise = self.discriminator_noise(noise_gen)
            confidence_noisy = self.discriminator_noisy(noisy_gen)
            confidence_speech = self.discriminator_speech(speech_gen)

            # generator_loss=generator_loss
            confidence_loss_speech = self.generator_loss(label=y_labels_1, fake_pred=confidence_speech) 
            confidence_loss_noisy = self.generator_loss(label=y_labels_2, fake_pred=confidence_noisy) 
            confidence_loss_noise = self.generator_loss(label=y_labels_3, fake_pred=confidence_noise)
            Confidence_loss = confidence_loss_speech + confidence_loss_noisy + confidence_loss_noise
            
            enhanced_speech_info = tf.concat([speech_gen, noisy_phase], axis=-1)
            clean_speech_info = tf.concat([clean_speech, clean_phase], axis=-1)
            enhanced_noisy_info = tf.concat([noisy_gen, noisy_phase], axis=-1)
            noisy_speech_info = tf.concat([noisy_speech, noisy_phase], axis=-1)
            enhanced_noise_info = tf.concat([noise_gen, noisy_phase], axis=-1)
            noise_info = tf.concat([noise, noise_phase], axis=-1)

            # Generator_loss=loss_func  return alpha * loss_sisdr + beta * loss_mae + gamma * loss_mse
            confidence_loss = self.Generator_loss(y_true=clean_speech_info, y_pred=enhanced_speech_info, alpha=0.8, beta=0.1,
                                                  gamma=0.1, input_mean=None, input_std=None) + \
                              self.Generator_loss(y_true=noisy_speech_info, y_pred=enhanced_noisy_info, alpha=0.4, beta=0.3, gamma=0.3,
                                                  input_mean=None, input_std=None) + \
                              self.Generator_loss(y_true=noise_info, y_pred=enhanced_noise_info, alpha=0, beta=0.5, gamma=0.5,
                                                  input_mean=None, input_std=None) + Confidence_loss

            # Identity_loss=Identity_loss return wave_loss * 0.99 + mel_loss * 0.01
            recon_loss = tf.reduce_mean(self.reconstruction_loss(y_true=noisy_speech_info, y_pred=enhanced_noisy_info, if_cosine=True))
            id_loss = tf.reduce_mean(
                self.Identity_loss(y_true=clean_speech_info, y_pred=enhanced_speech_info, input_mean=None, input_std=None,
                                   if_cosine=True))
            gen_loss = confidence_loss + recon_loss + id_loss

            # Calculating and applying the gradients for generator
            grads_gen = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
            self.gen_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_weights))

            self.gen_loss_tracker.update_state(gen_loss)
            self.recon_loss_tracker.update_state(recon_loss)
            self.id_loss_tracker.update_state(id_loss)
            self.confidence_loss_tracker.update_state(confidence_loss)

            return {
                "gen_loss": self.gen_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "id_loss": self.id_loss_tracker.result(),
                "disc_loss": self.disc_loss_tracker.result(),
                "disc_speech_loss": self.disc_speech_tracker.result(),
                "disc_noisy_loss": self.disc_noisy_tracker.result(),
                "disc_noise_loss": self.disc_noise_tracker.result()
            }
        # implement the call method
    def call(self, inputs, *args, **kwargs):
        pass


# Optimizers & LR schedulers
lr_batch = end - start
first_gap =3
second_gap = 5
genlr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [lr_batch * first_gap, lr_batch * second_gap], [1e-5, 1e-6, 1e-7])
genwd_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [lr_batch * first_gap, lr_batch * second_gap], [1e-5, 1e-5, 1e-7])
dislr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [lr_batch * first_gap, lr_batch * second_gap], [2*1e-5, 2*1e-6, 2*1e-7])
diswd_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [lr_batch * first_gap, lr_batch * second_gap], [2*1e-5, 2*1e-6, 2*1e-7])

G_optimizer = AdamW(learning_rate=genlr_schedule, weight_decay=genwd_schedule, beta_1=0.5)
D_speech_optimizer = AdamW(learning_rate=dislr_schedule, weight_decay=diswd_schedule, beta_1=0.5)
D_noise_optimizer = AdamW(learning_rate=dislr_schedule, weight_decay=diswd_schedule, beta_1=0.5)
D_noisy_optimizer = AdamW(learning_rate=dislr_schedule, weight_decay=diswd_schedule, beta_1=0.5)

# mirrored_strategy = tf.distribute.MirroredStrategy()
# central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()
# multi_worker_strategy = tf.distribute.MultiWorkerMirroredStrategy()

# with central_storage_strategy.scope():
this_model = Jianqiao_GAN(generator=generator,
                            discriminator_speech=discriminator_speech,
                            discriminator_noisy=discriminator_noisy,
                            discriminator_noise=discriminator_noise)

this_model.new_compile(gen_optimizer=G_optimizer,
                        disc_speech_optimizer=D_speech_optimizer,
                        disc_noisy_optimizer=D_noisy_optimizer,
                        disc_noise_optimizer=D_noise_optimizer,
                        generator_loss=generator_loss,
                        Generator_loss=loss_func,
                        Identity_loss=Identity_loss,
                        reconstruction_loss=reconstruction_loss)
"""
def frame_to_mag(frames, win_len, step_len, if_split=False, if_emphasis=False, if_norm=False, power_compress=False):
    if if_split:
        clean_speech, noisy_speech = tf.split(frames, num_or_size_splits=2, axis=-1)
        #frames -> waves
        clean_wave = inverse_framing(clean_speech)
        noisy_wave = inverse_framing(noisy_speech)

        # emphasis
        if if_emphasis:
            pre_emphasis = 0.97
            clean_wave = tf.concat([clean_wave[0:1], clean_wave[1:] - pre_emphasis * clean_wave[:-1]], axis=0)
            noisy_wave = tf.concat([noisy_wave[0:1], noisy_wave[1:] - pre_emphasis * noisy_wave[:-1]], axis=0)

        # norm
        if if_norm:
            c = tf.sqrt(tf.cast(tf.shape(noisy_wave)[-1], tf.float32) / tf.reduce_sum(noisy_wave ** 2.0, axis=-1))
            clean_wave = tf.transpose(clean_wave, perm=[1, 0])
            noisy_wave = tf.transpose(noisy_wave, perm=[1, 0])
            clean_wave = tf.transpose(clean_wave * c, perm=[1, 0])
            noisy_wave = tf.transpose(noisy_wave * c, perm=[1, 0])
            # clean_wave, noisy_wave = tf.transpose(tf.transpose(clean_wave, perm=[1, 0]) * c, perm=[1, 0]), tf.transpose(tf.transpose(noisy_wave, perm=[1, 0]) * c, perm=[1, 0])

        # stft
        # clean_wave = tf.squeeze(clean_wave, axis=-1)
        # noisy_wave = tf.squeeze(noisy_wave, axis=-1)
        clean_stft = tf.signal.stft(clean_wave, win_len, step_len, fft_length=win_len)
        noisy_stft = tf.signal.stft(noisy_wave, win_len, step_len, fft_length=win_len)
        clean_mag = tf.abs(clean_stft)
        noisy_mag = tf.abs(noisy_stft)
        if power_compress:
            clean_mag = tf.pow(clean_mag, 0.3)
            noisy_mag = tf.pow(noisy_mag, 0.3)
            
        return clean_mag, noisy_mag
    else:
        wave = inverse_framing(frames)
        if if_emphasis:
            pre_emphasis = 0.97
            wave = tf.concat([wave[0:1], wave[1:] - pre_emphasis * wave[:-1]], axis=0)
        # wave = tf.squeeze(wave, axis=-1)
        stft = tf.signal.stft(wave, win_len, step_len, fft_length=win_len)
        mag = tf.abs(stft)
        return mag

def frames_to_mags(frames, win_len, step_len, if_split=False):
    mags = []
    if if_split==False:
        for i in frames:
            mags.append(tf.expand_dims(frame_to_mag(i, win_len, step_len, if_split), axis=0))
        return tf.concat(mags, axis=0)
    else:
        for i in frames:
            clean_mag, noisy_mag = frame_to_mag(i, win_len, step_len, if_split)
            union = tf.concat([clean_mag, noisy_mag], axis=-1)
            mags.append(tf.expand_dims(union, axis=0))
        return tf.concat(mags, axis=0)
"""
def frame_to_mag(frames, win_len, step_len, if_split=False, if_emphasis=False, if_norm=True, power_compress=False):
    """
    input: frames
    :param frames:
    :param win_len:
    :param step_len:
    :param if_split:
    :param if_emphasis:
    :param if_norm:
    :param power_compress:
    :return: tf.concat([mag, phase], axis=-1)
    """
    if if_split:
        clean_speech, noisy_speech = tf.split(frames, num_or_size_splits=2, axis=-1)
        #frames -> waves
        clean_wave = inverse_framing(clean_speech)
        noisy_wave = inverse_framing(noisy_speech)

        # emphasis
        if if_emphasis:
            pre_emphasis = 0.97
            clean_wave = tf.concat([clean_wave[0:1], clean_wave[1:] - pre_emphasis * clean_wave[:-1]], axis=0)
            noisy_wave = tf.concat([noisy_wave[0:1], noisy_wave[1:] - pre_emphasis * noisy_wave[:-1]], axis=0)

        # norm
        if if_norm:
            c = tf.sqrt(tf.cast(tf.shape(noisy_wave)[-1], tf.float32) / tf.reduce_sum(noisy_wave ** 2.0, axis=-1))
            clean_wave = clean_wave * c
            noisy_wave = noisy_wave * c
            # clean_wave = tf.transpose(clean_wave, perm=[1, 0])
            # noisy_wave = tf.transpose(noisy_wave, perm=[1, 0])
            # clean_wave = tf.transpose(clean_wave * c, perm=[1, 0])
            # noisy_wave = tf.transpose(noisy_wave * c, perm=[1, 0])
            # clean_wave, noisy_wave = tf.transpose(tf.transpose(clean_wave, perm=[1, 0]) * c, perm=[1, 0]), tf.transpose(tf.transpose(noisy_wave, perm=[1, 0]) * c, perm=[1, 0])

        # stft
        # clean_wave = tf.squeeze(clean_wave, axis=-1)
        # noisy_wave = tf.squeeze(noisy_wave, axis=-1)
        clean_stft = tf.signal.stft(clean_wave, win_len, step_len, fft_length=win_len)
        noisy_stft = tf.signal.stft(noisy_wave, win_len, step_len, fft_length=win_len)
        clean_mag = tf.abs(clean_stft)
        noisy_mag = tf.abs(noisy_stft)
        clean_phase = tf.math.angle(clean_stft)
        noisy_phase = tf.math.angle(noisy_stft)
        if power_compress:
            clean_mag = tf.pow(clean_mag, 0.3)
            noisy_mag = tf.pow(noisy_mag, 0.3)

        return tf.concat([clean_mag, clean_phase], axis=-1), tf.concat([noisy_mag, noisy_phase], axis=-1)
    else:
        wave = inverse_framing(frames)
        if if_emphasis:
            pre_emphasis = 0.97
            wave = tf.concat([wave[0:1], wave[1:] - pre_emphasis * wave[:-1]], axis=0)
        # wave = tf.squeeze(wave, axis=-1)
        if if_norm:
            c = tf.sqrt(tf.cast(tf.shape(wave)[-1], tf.float32) / tf.reduce_sum(wave ** 2.0, axis=-1))
            wave = wave * c
            # wave = tf.transpose(wave, perm=[1, 0])
            # wave = tf.transpose(wave * c, perm=[1, 0])
            # wave = tf.transpose(tf.transpose(wave, perm=[1, 0]) * c, perm=[1, 0])
        stft = tf.signal.stft(wave, win_len, step_len, fft_length=win_len)
        mag = tf.abs(stft)
        phase = tf.math.angle(stft)
        return mag, phase

def frames_to_mags(frames, win_len, step_len, if_split=False):

    if if_split==False:
        mags = []
        phases = []
        for i in frames:
            mag, phase = frame_to_mag(i, win_len, step_len, if_split)
            mags.append(tf.expand_dims(mag, axis=0))
            phases.append(tf.expand_dims(phase, axis=0))
        mags = tf.concat(mags, axis=0)
        phases = tf.concat(phases, axis=0)
        return tf.concat([mags, phases], axis=-1)
    else:
        clean_mags = []
        clean_phases = []
        noisy_mags = []
        noisy_phases = []
        for i in frames:
            # clean_info, noisy_info = tf.split(i, num_or_size_splits=2, axis=-1)
            # clean_mag, clean_phase = frame_to_mag(clean_info, win_len, step_len, if_split)
            # noisy_mag, noisy_phase = frame_to_mag(noisy_info, win_len, step_len, if_split)
            clean_info, noisy_info = frame_to_mag(i, win_len, step_len, if_split)
            clean_mag, clean_phase = tf.split(clean_info, num_or_size_splits=2, axis=-1)
            noisy_mag, noisy_phase = tf.split(noisy_info, num_or_size_splits=2, axis=-1)

            clean_mags.append(tf.expand_dims(clean_mag, axis=0))
            clean_phases.append(tf.expand_dims(clean_phase, axis=0))
            noisy_mags.append(tf.expand_dims(noisy_mag, axis=0))
            noisy_phases.append(tf.expand_dims(noisy_phase, axis=0))

        clean_mags = tf.concat(clean_mags, axis=0)
        clean_phases = tf.concat(clean_phases, axis=0)
        noisy_mags = tf.concat(noisy_mags, axis=0)
        noisy_phases = tf.concat(noisy_phases, axis=0)

        clean_speech = tf.concat([clean_mags, clean_phases], axis=-1)
        noisy_speech = tf.concat([noisy_mags, noisy_phases], axis=-1)
        return tf.concat([clean_speech, noisy_speech], axis=-1)
        

checkpoint_save_path = "./1st_model_param/model.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    this_model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                save_weights_only=True,
                                                save_best_only=True)

with tf.device('/cpu:0'):
    
    base_path = '/scratch/jc18n17/data/subjective_experiment/pre_train_waveform/'

    tfrecord_train_X = base_path + '1st_Train_X.txt'  
    tfrecord_test_X = base_path + '1st_Test_X.txt'  
    tfrecord_train_y = base_path + '1st_Train_y.txt'  
    tfrecord_test_y = base_path + '1st_Test_y.txt'  

    np_train_X = pickle.load(open(tfrecord_train_X, 'rb'))
    np_test_X = pickle.load(open(tfrecord_test_X, 'rb'))
    np_train_y = pickle.load(open(tfrecord_train_y, 'rb'))
    np_test_y = pickle.load(open(tfrecord_test_y, 'rb'))

    np_train_X = np.array(np_train_X, dtype=np.float32)
    np_train_y = np.array(np_train_y, dtype=np.float32)
    np_test_X = np.array(np_test_X, dtype=np.float32)
    np_test_y = np.array(np_test_y, dtype=np.float32)

    divide_rate = 0.2
    np_train_X = np_train_X[3 * int(len(np_train_X) * divide_rate): 4 * int(len(np_train_X) * divide_rate)]
    np_train_y = np_train_y[3 * int(len(np_train_y) * divide_rate):4 * int(len(np_train_y) * divide_rate)]
    np_test_X = np_test_X[3 * int(len(np_test_X) * divide_rate):4 * int(len(np_test_X) * divide_rate)]
    np_test_y = np_test_y[3 * int(len(np_test_y) * divide_rate):4 * int(len(np_test_y) * divide_rate)]
    
    np_train_X = frames_to_mags(np_train_X, win_len=510, step_len=255, if_split=True)
    np_test_X = frames_to_mags(np_test_X, win_len=510, step_len=255, if_split=True)
    np_train_y = frames_to_mags(np_train_y, win_len=510, step_len=255, if_split=False)
    np_test_y = frames_to_mags(np_test_y, win_len=510, step_len=255, if_split=False)

    # rate = 0.3
    # train_size = int(len(np_train_X) * rate)
    # test_size = int(len(np_test_X) * rate)

    # round = 3
    # np_train_X = np_train_X[round * train_size: (round+1) *train_size]
    # np_train_y = np_train_y[round *train_size:(round+1)*train_size]
    # np_test_X = np_test_X[round *test_size:(round+1)*test_size]
    # np_test_y = np_test_y[round *test_size:(round+1)*test_size]
    print(np_train_X.shape)
    print(np_train_y.shape)
    print(np_test_X.shape)
    print(np_test_y.shape)

    
with tf.device('/gpu:0'):
    from tensorflow.keras.utils import Sequence 
    class DataGenerator(Sequence):
        def __init__(self, x_set, y_set, batch_size):
            # x_set = frames_to_mags(x_set, 510, 255, if_split=True)
            # y_set = frames_to_mags(y_set, 510, 255, if_split=False)
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self.x) / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

    train_gen = DataGenerator(np_train_X, np_train_y, 1) # default batch_size = 2, 当你要修改bātch_size的时候，记得修改ATFT.py和AHA.py中的batch_size，分别是它们对应的可训练参数，否则会报错！！！！！！！
    test_gen = DataGenerator(np_test_X, np_test_y, 1)# default batch_size = 2, 当你要修改bātch_size的时候，记得修改ATFT.py和AHA.py中的batch_size，分别是它们对应的可训练参数，否则会报错！！！！！！！

    history = this_model.fit_generator(train_gen,
                        epochs=epochs,
                        validation_data=test_gen,
                        # validation_freq=1,
                        callbacks=[cp_callback])
exit()


with tf.device('/gpu:0'):
    history = this_model.fit(np_train_X, np_train_y, batch_size=batch_size, epochs=epochs, validation_data=(np_test_X, np_test_y), validation_freq=1,
                    callbacks=[cp_callback])
    # ßhistory = this_model.fit(dataset_train, epochs=epochs, validation_data=dataset_test, validation_freq=1, callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs'),cp_callback])
                        # callbacks=[cp_callback])
                        # callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs'),cp_callback])



    # draw loss function
plt.plot(history.history['gen_loss'])
plt.plot(history.history['val_gen_loss'])
gen_loss_median = np.median(history.history['gen_loss'])
val_gen_loss_median = np.median(history.history['val_gen_loss'])
plt.axhline(y=gen_loss_median, color='r', linestyle='-')
plt.axhline(y=val_gen_loss_median, color='g', linestyle='-')
plt.title('Generator Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train_gen', 'Validation_gen'], loc='upper right')
plt.show()
# Save the figure
plt.savefig('gen_loss.png')

plt.plot(history.history['disc_loss'])
plt.plot(history.history['val_disc_loss'])
disc_loss_median = np.median(history.history['disc_loss'])
val_disc_loss_median = np.median(history.history['val_disc_loss'])
plt.axhline(y=disc_loss_median, color='r', linestyle='-')
plt.axhline(y=val_disc_loss_median, color='g', linestyle='-')
plt.title('Discriminator Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train_disc', 'Validation_disc'], loc='upper right')
plt.show()
# Save the figure
plt.savefig('disc_loss.png')
