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
        ######################CIFAR
        cifar = CIFAR()
        print(cifar.test_data.shape)

        tfparams = np.load('tf_ood_params.npy')
        print(tfparams.shape)

        k = 0
        for tfparam in tfparams:
            # print(tfparam.shape)

            # 原版cifar经裁剪 加仿射变换
            tform = transform.AffineTransform(tfparam)
            # (10000, 32, 32, 3) -> (10000, 28, 28, 1)
            reshaped_cifar = [transform.warp(x[2:30, 2:30, 0], tform) for x in cifar.test_data]
            reshaped_cifar = np.reshape(reshaped_cifar, (10000, 28, 28, 1))

            print(np.array(reshaped_cifar).shape)
            cifar_test_result = model.model.predict(reshaped_cifar)

            np.save('exp_affine_cifar_%d.npy' % k, cifar_test_result)
            k += 1
            print(cifar_test_result.shape)
