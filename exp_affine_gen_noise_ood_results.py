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
        ###################### noise
        # cifar = CIFAR()
        # print(cifar.test_data.shape)
        np.random.seed(1234)
        noise = np.random.rand(10000, 28, 28, 1)

        tfparams = np.load('tf_ood_params.npy')
        print(tfparams.shape)

        k = 0
        for tfparam in tfparams:
            # print(tfparam.shape)

            # 原版cifar经裁剪 加仿射变换
            tform = transform.AffineTransform(tfparam)
            # (10000, 32, 32, 3) -> (10000, 28, 28, 1)
            reshaped_noise = [transform.warp(x, tform) for x in noise]
            reshaped_noise = np.reshape(reshaped_noise, (10000, 28, 28, 1))
            reshaped_noise = reshaped_noise - 0.5

            # print(np.array(reshaped_noise).shape)
            noise_test_result = model.model.predict(reshaped_noise)

            np.save('exp_affine_noise_%d.npy' % k, noise_test_result)
            k += 1
            print(noise_test_result.shape)
