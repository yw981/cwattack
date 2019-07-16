import tensorflow as tf
import numpy as np
from skimage import filters, transform

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel


def eval(label, arr):
    print(label + " count ", arr.shape[0], " max ", np.max(arr), " min ", np.min(arr), " mean ", np.mean(arr), " var ",
          np.var(arr), " median ", np.median(arr))


# 测试
if __name__ == "__main__":
    with tf.Session() as sess:
        model = MNISTModel("models/mnist", sess)
        # mnist = MNIST()
        # cifar = CIFAR()
        # eval('mnist',mnist.test_data)
        # eval('cifar',cifar.test_data)

        # mnist = MNIST()
        # # 原版mnist 加高斯过滤
        # # sigma = 0.5
        # # variant_mnist = [filters.gaussian(x, sigma) for x in mnist.test_data]
        #
        # # 原版mnist 加仿射变换 旋转角度
        # # tform = transform.AffineTransform(rotation=-3.14 / 24)  # 逆时针旋转，弧度，绕左上角顶点
        # # 原版mnist 加仿射变换 平移
        # tform = transform.AffineTransform(translation=(2, 2))  # 平移（x,y) x正水平向右，y正竖直向上
        # # 原版mnist 加仿射变换 缩放
        # # tform = transform.AffineTransform(scale=(1.1, 1.1))
        # variant_mnist = [transform.warp(x, tform) for x in mnist.test_data]
        # variant_mnist = np.reshape(variant_mnist, (10000, 28, 28, 1))
        #
        # # print(np.array(reshaped_cifar).shape)
        # mnist_test_result = model.model.predict(variant_mnist)
        #
        # np.save('test_result_translation.npy', mnist_test_result)
        # print(mnist_test_result.shape)




        #######################CIFAR
        # cifar = CIFAR()
        #
        # print(cifar.test_data.shape)
        # # (10000, 32, 32, 3) -> (10000, 28, 28, 1)
        # # 原版cifar经裁剪
        # # reshaped_cifar = [x[2:30, 2:30, 0] for x in cifar.test_data]
        #
        # # 原版cifar经裁剪 加高斯过滤
        # # sigma = 0.5
        # # reshaped_cifar = [filters.gaussian(x[2:30, 2:30, 0], sigma) for x in cifar.test_data]
        #
        # # 原版cifar经裁剪 加仿射变换 旋转角度
        # # tform = transform.AffineTransform(rotation=-3.14 / 24)  # 逆时针旋转，弧度，绕左上角顶点
        # # 原版cifar经裁剪 加仿射变换 平移
        # tform = transform.AffineTransform(translation=(2, 2))  # 平移（x,y) x正水平向右，y正竖直向上
        # # 原版cifar经裁剪 加仿射变换 缩放
        # # tform = transform.AffineTransform(scale=(1.1, 1.1))
        # reshaped_cifar = [transform.warp(x[2:30, 2:30, 0], tform) for x in cifar.test_data]
        #
        # reshaped_cifar = np.reshape(reshaped_cifar, (10000, 28, 28, 1))
        #
        # # print(np.array(reshaped_cifar).shape)
        # cifar_test_result = model.model.predict(reshaped_cifar)
        #
        # np.save('cifar_translation_test_result.npy', cifar_test_result)
        # print(cifar_test_result.shape)

        #######################NOISE as OOD
        np.random.seed(1234)
        noise = np.random.rand(10000, 28, 28, 1) - 0.5
        # noise 加高斯过滤 注意：原数据∈[-0.5,0.5]会出错，加回去
        sigma = 0.5
        variant_noise = [filters.gaussian(x + 0.5, sigma) for x in noise]
        variant_noise = np.array(variant_noise) - 0.5
        #
        # noise 加仿射变换 旋转角度
        # tform = transform.AffineTransform(rotation=-3.14 / 24)  # 逆时针旋转，弧度，绕左上角顶点
        # noise 加仿射变换 平移
        # tform = transform.AffineTransform(translation=(2, 2))  # 平移（x,y) x正水平向右，y正竖直向上
        # noise 加仿射变换 缩放
        # tform = transform.AffineTransform(scale=(1.1, 1.1))
        # noise 加仿射变换 shear
        # tform = transform.AffineTransform(shear=3.14 / 24)
        # print(tform.params)
        # variant_noise = [transform.warp(x, tform) for x in noise]
        # variant_noise = np.reshape(variant_noise, (10000, 28, 28, 1))

        test_result = model.model.predict(variant_noise)

        np.save('noise_result_gaussian.npy', test_result)
        print(test_result.shape)
