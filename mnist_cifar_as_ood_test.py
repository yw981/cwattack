## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters, transform

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            # 共有seq个标签
            for j in seq:
                # 标签正确是自己时，不输出
                if (j == np.argmax(data.test_labels[start + i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def show_images(npdata, num_row=10):
    fig = plt.figure()
    plt.subplots_adjust(wspace=1, hspace=1)
    rows = npdata.shape[0] // num_row + 1  # 加一行对比
    for i in range(npdata.shape[0]):
        img = npdata[i].reshape([28, 28])
        posp = fig.add_subplot(rows, num_row, i + 1)
        # plt.title('%d : %d -> %d %.2f' % (i, labels[i], error_preds[i], error_scores[i]))
        posp.imshow(img, cmap=plt.cm.gray)

    for i in range(num_row):
        dif = npdata[i + num_row].reshape([28, 28]) - npdata[i].reshape([28, 28])
        posp = fig.add_subplot(rows, num_row, num_row * 2 + i + 1)
        posp.imshow(dif, cmap=plt.cm.gray)

    plt.show()


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def eval(label, arr):
    print(label + " count ", arr.shape[0], " max ", np.max(arr), " min ", np.min(arr), " mean ", np.mean(arr), " var ",
          np.var(arr), " median ", np.median(arr))


def compare_result(label, result):
    softmax_scores = np.max(softmax(result), axis=1)
    amr = np.argmax(result, axis=1)
    aml = np.argmax(data.test_labels, axis=1)
    # print(amr)
    # print(aml)
    wrong_indices = (amr != aml)
    right_indices = ~wrong_indices
    print("acc = %f" % (1 - np.sum(wrong_indices) / aml.shape[0]))
    right_softmax_scores = softmax_scores[right_indices]
    wrong_softmax_scores = softmax_scores[wrong_indices]
    eval(label + ' right', right_softmax_scores)
    eval(label + ' wrong', wrong_softmax_scores)

def compare_ood_result(label, result):
    cifar_softmax_scores = np.max(softmax(result), axis=1)
    eval(label, cifar_softmax_scores)

def compare_ood(label, origin,variant):
    amo = np.argmax(origin, axis=1)
    amv = np.argmax(variant, axis=1)
    print(amo)
    print(amv)
    error_indices = (amo != amv)
    print(label+" dif = %f" % (np.sum(error_indices) / amo.shape[0]))


# 测试
if __name__ == "__main__":
    data = MNIST()
    cifar = CIFAR()

    test_result = np.load('test_result.npy')
    test_result_gaussian = np.load('test_result_gaussian.npy')
    test_result_rotation = np.load('test_result_rotation.npy')
    compare_result('mnist', test_result)
    compare_result('mnist_gaussian', test_result_gaussian)
    compare_result('mnist_rotation', test_result_rotation)

    # np.savetxt('softmax_scores.npy', softmax_scores)
    cifar_test_result = np.load('cifar_test_result.npy')
    cifar_gaussian_test_result = np.load('cifar_gaussian_test_result.npy')
    cifar_rotation_test_result = np.load('cifar_rotation_test_result.npy')

    compare_ood_result('cifar',cifar_test_result)
    compare_ood_result('cifar_gaussian',cifar_gaussian_test_result)
    compare_ood_result('cifar_rotation',cifar_rotation_test_result)

    compare_ood('cifar_gaussian',cifar_test_result,cifar_gaussian_test_result)
    compare_ood('cifar_rotation',cifar_test_result,cifar_rotation_test_result)

    # 再做一组纯噪声

    # np.savetxt('cifar_softmax_scores.npy', cifar_softmax_scores)
