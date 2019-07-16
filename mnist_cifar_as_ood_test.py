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

###
# 比较两个结果集不一样的
###
def compare_ood(label, origin, variant):
    # argmax origin
    amo = np.argmax(origin, axis=1)
    # argmax variant
    amv = np.argmax(variant, axis=1)
    print(amo)
    print(amv)
    error_indices = (amo != amv)
    print(label + " dif = %f" % (np.sum(error_indices) / amo.shape[0]))


def count_diff(tag, origin, variant_arr):
    """
    对比变化，origin预测结果和variant结果数组中，只要有任意不同，就找剔除
    :param tag: 输出TAG
    :param origin: 原标签
    :param variant_arr: 变体标签数组
    :return:
    """
    re = np.ones(origin.shape[0]).astype(np.bool)

    for variant in variant_arr:
        tfs = (np.argmax(origin, axis=1) == np.argmax(variant, axis=1))
        # print(tfs)
        # re = np.bitwise_and(re,tfs)
        re = re & tfs

    # 输出都没变的
    print(tag, np.sum(re), ' out of ', re.shape[0], ' at rate ', np.sum(re) / re.shape[0])
    # return re


# 测试
if __name__ == "__main__":
    data = MNIST()
    cifar = CIFAR()
    #
    test_result = np.load('test_result.npy')
    test_result_gaussian = np.load('test_result_gaussian.npy')
    test_result_rotation = np.load('test_result_rotation.npy')
    test_result_scale = np.load('test_result_scale.npy')
    test_result_translation = np.load('test_result_translation.npy')
    compare_result('mnist', test_result)
    compare_result('mnist_gaussian', test_result_gaussian)
    compare_result('mnist_rotation', test_result_rotation)
    compare_result('mnist_scale', test_result_scale)
    compare_result('mnist_translation', test_result_translation)
    print('\n')

    # np.savetxt('softmax_scores.npy', softmax_scores)
    cifar_test_result = np.load('cifar_test_result.npy')
    cifar_gaussian_test_result = np.load('cifar_gaussian_test_result.npy')
    cifar_rotation_test_result = np.load('cifar_rotation_test_result.npy')
    cifar_scale_test_result = np.load('cifar_scale_test_result.npy')
    cifar_translation_test_result = np.load('cifar_translation_test_result.npy')

    compare_ood_result('cifar', cifar_test_result)
    compare_ood_result('cifar_gaussian', cifar_gaussian_test_result)
    compare_ood_result('cifar_rotation', cifar_rotation_test_result)
    compare_ood_result('cifar_scale', cifar_scale_test_result)
    compare_ood_result('cifar_translation', cifar_translation_test_result)
    print('\n')

    compare_ood('cifar_gaussian', cifar_test_result, cifar_gaussian_test_result)
    compare_ood('cifar_rotation', cifar_test_result, cifar_rotation_test_result)
    compare_ood('cifar_scale', cifar_test_result, cifar_scale_test_result)
    compare_ood('cifar_translation', cifar_test_result, cifar_translation_test_result)
    print('\n')

    # 再做一组纯噪声
    noise_test_result = np.load('noise_result.npy')
    noise_rotation_test_result = np.load('noise_result_rotation.npy')
    noise_scale_test_result = np.load('noise_result_scale.npy')
    noise_translation_test_result = np.load('noise_result_translation.npy')
    compare_ood_result('noise', noise_test_result)
    compare_ood_result('noise_rotation', noise_rotation_test_result)
    compare_ood_result('noise_scale', noise_scale_test_result)
    compare_ood_result('noise_translation', noise_translation_test_result)
    print('\n')
    compare_ood('noise_rotation', noise_test_result, noise_rotation_test_result)
    compare_ood('noise_scale', noise_test_result, noise_scale_test_result)
    compare_ood('noise_translation', noise_test_result, noise_translation_test_result)
    print('\n')

    count_diff('mnist_in', test_result,
               [test_result_gaussian, test_result_rotation, test_result_scale, test_result_translation])

    count_diff('cifar_ood', cifar_test_result,
               [cifar_gaussian_test_result, cifar_rotation_test_result, cifar_scale_test_result,
                cifar_translation_test_result])

    count_diff('noise_ood', noise_test_result,
               [noise_rotation_test_result, noise_scale_test_result,
                noise_translation_test_result])





    # np.savetxt('cifar_softmax_scores.npy', cifar_softmax_scores)
