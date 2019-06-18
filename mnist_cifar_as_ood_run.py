import tensorflow as tf
import numpy as np
from skimage import filters, transform

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel

# 测试
if __name__ == "__main__":
    with tf.Session() as sess:
        model = MNISTModel("models/mnist", sess)

        # mnist = MNIST()
        # # 原版mnist 加高斯过滤
        # # sigma = 0.5
        # # variant_mnist = [filters.gaussian(x, sigma) for x in mnist.test_data]
        #
        # # 原版mnist 加仿射变换 旋转角度
        # tform = transform.AffineTransform(rotation=-3.14 / 24)  # 逆时针旋转，弧度，绕左上角顶点
        # variant_mnist = [transform.warp(x, tform) for x in mnist.test_data]
        #
        # variant_mnist = np.reshape(variant_mnist, (10000, 28, 28, 1))
        #
        # # print(np.array(reshaped_cifar).shape)
        # mnist_test_result = model.model.predict(variant_mnist)
        #
        # np.save('test_result_rotation.npy', mnist_test_result)
        # print(mnist_test_result.shape)




        #######################CIFAR
        cifar = CIFAR()

        print(cifar.test_data.shape)
        # (10000, 32, 32, 3) -> (10000, 28, 28, 1)
        # 原版cifar经裁剪
        # reshaped_cifar = [x[2:30, 2:30, 0] for x in cifar.test_data]

        # 原版cifar经裁剪 加高斯过滤
        # sigma = 0.5
        # reshaped_cifar = [filters.gaussian(x[2:30, 2:30, 0], sigma) for x in cifar.test_data]

        # 原版cifar经裁剪 加仿射变换 旋转角度
        # tform = transform.AffineTransform(rotation=-3.14 / 24)  # 逆时针旋转，弧度，绕左上角顶点
        # 原版cifar经裁剪 加仿射变换 平移
        # tform = transform.AffineTransform(translation=(1, 0))  # 平移（x,y) x正水平向右，y正竖直向上
        # 原版cifar经裁剪 加仿射变换 缩放
        tform = transform.AffineTransform(scale=(0.98, 0.98))
        reshaped_cifar = [transform.warp(x[2:30, 2:30, 0], tform) for x in cifar.test_data]

        reshaped_cifar = np.reshape(reshaped_cifar, (10000, 28, 28, 1))

        # print(np.array(reshaped_cifar).shape)
        cifar_test_result = model.model.predict(reshaped_cifar)

        np.save('cifar_scale_test_result.npy', cifar_test_result)
        print(cifar_test_result.shape)
