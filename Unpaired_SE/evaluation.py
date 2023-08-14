import tensorflow as tf

def si_sdr(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # 去掉无用的维度
    # y_true表示原始的语音信号，因此其shape应为(batch_size, time_steps, num_channels， 1)
    if len(y_true.shape) == 3 and y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
    # 计算目标信号和噪声信号
    e_n = y_true - tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True) * y_pred / tf.reduce_sum( y_pred * y_pred, axis=-1, keepdims=True)
    # 计算 SDR
    e_sn = tf.reduce_sum(y_true**2, axis=-1) / (tf.reduce_sum(e_n**2, axis=-1) + 1e-10)
    # 转换为负的 SDR 平均值
    return -tf.reduce_mean(10 * tf.math.log(e_sn) / tf.math.log(10.0))

def tf_si_sdr(reference, estimation):
    # 强制转换 reference 和 estimation 为相同的数据类型（例如，tf.float32）,输入的特征一定要是时域特征也就是waveform
    reference = tf.cast(reference, tf.float32)
    estimation = tf.cast(estimation, tf.float32)

    estimation, reference = tf.broadcast_to(estimation, tf.shape(reference)), tf.broadcast_to(reference, tf.shape(estimation))

    reference_energy = tf.reduce_sum(reference ** 2, axis=-1, keepdims=True)

    optimal_scaling = tf.reduce_sum(reference * estimation, axis=-1, keepdims=True) \
        / reference_energy
    projection = optimal_scaling * reference
    noise = estimation - projection

    ratio = tf.reduce_sum((projection + 1e-8) ** 2, axis=-1) / tf.reduce_sum((noise + 1e-8) ** 2, axis=-1)
    return 10 * tf.math.log(ratio) / tf.math.log(10.0)

def sisdr(y_true, y_pred):
    # y_true和y_pred的形状均为(batch_size, num_samples, num_channels)
    eps = 1e-8
    batch_size = tf.shape(y_true)[0]
    num_samples = tf.shape(y_true)[1]
    num_channels = tf.shape(y_true)[2]
    # y_true = tf.cast(y_true, dtype=tf.float32)
    # y_pred = tf.cast(y_pred, dtype=tf.float32)

    # 计算源信号和估计信号之间的内积
    dot = tf.reduce_sum(y_true * y_pred, axis=1)
    # 计算源信号的方差
    s_true = tf.reduce_sum(y_true ** 2, axis=1)
    # 计算估计信号和源信号之间的比例因子
    alpha = dot / (s_true + eps)
    # 对估计信号进行缩放，使其和源信号尽量接近
    y_pred_scaled = alpha[:, tf.newaxis, :] * y_true
    # 计算残差信号
    e = y_pred - y_pred_scaled
    # 计算源信号的方差
    s_true = tf.reduce_sum(y_true ** 2, axis=1)
    # 计算残差信号的方差
    e_var = tf.reduce_sum(e ** 2, axis=1)
    # 计算SiSDR
    sisdr = 10 * tf.math.log(s_true / (e_var + eps) + eps) / tf.math.log(10.0)
    # 求平均值
    mean_sisdr = tf.reduce_mean(sisdr, axis=0)
    # 返回平均SiSDR
    return -mean_sisdr
