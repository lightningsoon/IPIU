import tensorflow as tf
from resnet_base import softmax_layer, conv_layer, residual_block

n_dict = {20: 1, 32: 2, 44: 3, 56: 4}


# ResNet architectures used for CIFAR-10
# 这个n能确定生成的层数
# 每个小模块算3层
# 1+num_conv*2*2*3+1
def resnet(inpt, n):
    if n < 20 or (n - 20) % 12 != 0:
        print
        "ResNet depth invalid."
        return

    num_conv = (n - 20) / 12 + 1
    layers = []

    # 1.输入卷积
    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 3, 16], 1)
        layers.append(conv1)
    # 2.两个无池化小模块
    for i in range(num_conv):
        with tf.variable_scope('conv2_%d' % (i + 1)):
            conv2_x = residual_block(layers[-1], 16, False)
            conv2 = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape().as_list()[1:] == [32, 32, 16]
    # 3.第一个一个池化小模块+后面的无池化
    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i + 1)):
            conv3_x = residual_block(layers[-1], 32, down_sample)
            conv3 = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [16, 16, 32]
    # 只有第一个池化，深度64个节点
    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i + 1)):
            conv4_x = residual_block(layers[-1], 64, down_sample)
            conv4 = residual_block(conv4_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [8, 8, 64]
    # 全连接一层
    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [64]

        out = softmax_layer(global_pool, [64, 10])
        layers.append(out)

    return layers[-1]