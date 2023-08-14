import tensorflow as tf

def inverse_framing(sig, frame_shift=128):
    return tf.signal.overlap_and_add(signal=sig, frame_step=frame_shift, name=None)
    
def si_sdr(y_true, y_pred):
    EPS = 1e-8
    # 去掉无用的维度
    # y_true表示原始的语音信号，因此其shape应为(batch_size, time_steps, num_channels， 1)
    if len(y_true.shape) == 3 and y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
    # 计算目标信号和噪声信号
    pair_wised_proj = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True) * y_true / (tf.reduce_sum( y_true * y_true, axis=-1, keepdims=True) + EPS)
    e_n = y_pred - pair_wised_proj
    # 计算 SDR
    e_sdr = tf.reduce_sum(pair_wised_proj**2, axis=-1) / (tf.reduce_sum(e_n**2, axis=-1) + EPS)
    # 转换为负的 SDR 平均值
    # SI_SDR = -(10 * tf.math.log(tf.maximum(e_sdr, EPS)) / tf.math.log(10.0))
    SI_SDR = -(tf.math.log(tf.maximum(e_sdr, EPS)) / tf.math.log(10.0))
    return SI_SDR

