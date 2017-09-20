import tensorflow as tf
import numpy as np


# 初始化权重
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def softmax_layer(inpt, shape):
    # 全连接层
    # 权重
    fc_w = weight_variable(shape)
    # 偏置参数
    fc_b = tf.Variable(tf.zeros([shape[1]]))
    # 计算概率，一个指数型数学公式
    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h

# 输入数据，神经元尺寸，步长
# 输出搭好运算图的网络结构，就是激活后的结果
def conv_layer(inpt, filter_shape, stride):
    # 0123维分别是 行 列 输入 输出
    out_channels = filter_shape[3]
    # 生成变量
    filter_ = weight_variable(filter_shape)
    # 求卷积
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    #
    mean, var = tf.nn.moments(conv, axes=[0, 1, 2])

    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    # batch_norm = tf.nn.batch_norm_with_global_normalization(
    #     conv, mean, var, beta, gamma, 0.001,
    #     scale_after_normalization=True)

    # 使参数分布更均匀，对效果有一定提高
    batch_norm = tf.nn.batch_normalization(conv, mean, var, beta, gamma, 0.001)
    out = tf.nn.relu(batch_norm)

    return out

# 一个个小模块
def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    # 下采样开=最大值池化
    if down_sample:
        filter_ = [1, 2, 2, 1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    # 输入x经过两个卷积，这是一个小型模块
    # 如果你看过网络模型，就会发现，ResNet正是由这样一个个小block
    # 组成哒！
    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)
    # 所有维想相加，那维度得一样
    # 如果不一样
    if input_depth != output_depth:
        # 巧妙的方法1*1卷积核不改变内容，只改变输入输出维度
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)

        # 参数表示为上下左右前后浅深补零的个数
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt
    # 可以加了
    res = conv2 + input_layer
    return res
