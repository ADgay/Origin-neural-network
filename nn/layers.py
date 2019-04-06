# -*- coding: utf-8 -*-
"""
Created on 2019/04/06

@author: 徐伟祝/Xu Weizhu

"""
import numpy as np


def fc_forward(z, W, b):
    """
    全连接层的前向传播
    :param z: 当前层的输出,形状 (N,ln)
    :param W: 当前层的权重
    :param b: 当前层的偏置
    :return: 下一层的输出
    """
    return np.dot(z, W) + b


def fc_backward(next_dz, W, z):
    """
    全连接层的反向传播
    :param next_dz: 下一层的梯度
    :param W: 当前层的权重
    :param z: 当前层的输出
    :return:
    """
    N = z.shape[0]
    dz = np.dot(next_dz, W.T)  # 当前层的梯度
    dw = np.dot(z.T, next_dz)  # 当前层权重的梯度
    db = np.sum(next_dz, axis=0)  # 当前层偏置的梯度, N个样本的梯度求和
    return dw / N, db / N, dz

def main():
    z = np.ones((5, 5))
    k = np.ones((3, 3))
    b = 3
    # print(_single_channel_conv(z, k,padding=(1,1)))
    # print(_single_channel_conv(z, k, strides=(2, 2)))
    assert _single_channel_conv(z, k).shape == (3, 3)
    assert _single_channel_conv(z, k, padding=(1, 1)).shape == (5, 5)
    assert _single_channel_conv(z, k, strides=(2, 2)).shape == (2, 2)
    assert _single_channel_conv(z, k, strides=(2, 2), padding=(1, 1)).shape == (3, 3)
    assert _single_channel_conv(z, k, strides=(2, 2), padding=(1, 0)).shape == (3, 2)
    assert _single_channel_conv(z, k, strides=(2, 1), padding=(1, 1)).shape == (3, 5)

    dz = np.ones((1, 1, 3, 3))
    assert _insert_zeros(dz, (1, 1)).shape == (1, 1, 3, 3)
    print(_insert_zeros(dz, (3, 2)))
    assert _insert_zeros(dz, (1, 2)).shape == (1, 1, 3, 5)
    assert _insert_zeros(dz, (2, 2)).shape == (1, 1, 5, 5)
    assert _insert_zeros(dz, (2, 4)).shape == (1, 1, 5, 9)

    z = np.ones((8, 16, 5, 5))
    k = np.ones((16, 32, 3, 3))
    b = np.ones((32))
    assert conv_forward(z, k, b).shape == (8, 32, 3, 3)
    print(conv_forward(z, k, b)[0, 0])

    print(np.argmax(np.array([[1, 2], [3, 4]])))


def test_conv():
    # 测试卷积
    z = np.random.randn(3, 3, 28, 28).astype(np.float64)
    K = np.random.randn(3, 4, 3, 3).astype(np.float64) * 1e-3
    b = np.zeros(4).astype(np.float64)

    next_z = conv_forward(z, K, b)
    y_true = np.ones_like(next_z)

    from nn.losses import mean_squared_loss
    for i in range(10000):
        # 前向
        next_z = conv_forward(z, K, b)
        # 反向
        loss, dy = mean_squared_loss(next_z, y_true)
        dK, db, _ = conv_backward(dy, K, z)
        K -= 0.001 * dK
        b -= 0.001 * db

        if i % 10 == 0:
            print("i:{},loss:{},mindy:{},maxdy:{}".format(i, loss, np.mean(dy), np.max(dy)))

        if np.allclose(y_true, next_z):
            print("yes")
            break

def test_conv_and_max_pooling():
    # 测试卷积和最大池化
    z = np.random.randn(3, 3, 28, 28).astype(np.float64)
    K = np.random.randn(3, 4, 3, 3).astype(np.float64) * 1e-3
    b = np.zeros(4).astype(np.float64)

    next_z = conv_forward(z, K, b)
    y_pred = max_pooling_forward_bak(next_z,pooling=(2,2))
    y_true = np.ones_like(y_pred)

    from nn.losses import mean_squared_loss
    for i in range(10000):
        # 前向
        next_z = conv_forward(z, K, b)
        y_pred = max_pooling_forward_bak(next_z, pooling=(2, 2))
        # 反向
        loss, dy = mean_squared_loss(y_pred, y_true)
        next_dz = max_pooling_backward_bak(dy,next_z,pooling=(2,2))
        dK, db, _ = conv_backward(next_dz, K, z)
        K -= 0.001 * dK
        b -= 0.001 * db

        if i % 10 == 0:
            print("i:{},loss:{},mindy:{},maxdy:{}".format(i, loss, np.mean(dy), np.max(dy)))

        if np.allclose(y_true, y_pred):
            print("yes")
            break

if __name__ == "__main__":
    main()
