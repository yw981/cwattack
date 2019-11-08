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

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import roc_curve, auc


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


    # 只有1和0的结果
    re = np.ones(origin.shape[0]).astype(np.bool)
    for variant in variant_arr:
        tfs = (np.argmax(origin, axis=1) == np.argmax(variant, axis=1))
        re = re & tfs


    # 输出都没变的
    print(tag, np.sum(re), ' out of ', re.shape[0], ' at rate ', np.sum(re) / re.shape[0])
    # return re


def cal_pr(y_score,y_test):
    # print(y_test)
    # print(y_score)

    average_precision = average_precision_score(y_test, y_score)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.show()


def cal_roc(y_score,y_test):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area，Numpy ravel()相当于flatten，内存引用不同
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # tpr fpr都是dict，0,1,2，micro
    # print(tpr)
    # print(fpr)


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def vis_diff(tag, in_origin, in_variant_arr, out_origin, out_variant_arr):
    """
    对比变化，origin预测结果和variant结果数组中，只要有任意不同，就找剔除
    :param tag: 输出TAG
    :param origin: 原标签
    :param variant_arr: 变体标签数组
    :return:
    """


    # 只有1和0的结果
    # re = np.ones(origin.shape[0]).astype(np.bool)
    # for variant in variant_arr:
    #     tfs = (np.argmax(origin, axis=1) == np.argmax(variant, axis=1))
    #     re = re & tfs

    # 连续得分的结果，不变得分，变了不得分
    in_re = np.zeros(in_origin.shape[0])
    length = len(in_variant_arr)
    for variant in in_variant_arr:
        tfs = (np.argmax(in_origin, axis=1) == np.argmax(variant, axis=1))
        tfs = tfs / length
        in_re = in_re + tfs

    # print(in_re)

    # 连续得分的结果，不变得分，变了不得分
    out_re = np.zeros(out_origin.shape[0])
    length = len(out_variant_arr)
    for variant in out_variant_arr:
        tfs = (np.argmax(out_origin, axis=1) == np.argmax(variant, axis=1))
        tfs = tfs / length
        out_re = out_re + tfs

    # print(out_re)
    # print(in_re.shape)
    # print(out_re.shape)


    in_label = np.ones(in_re.shape)
    out_label = np.zeros(out_re.shape)

    y_label = np.hstack((in_label,out_label))
    re = np.hstack((in_re,out_re))
    # print(re.shape)
    # print(y_label.shape)
    cal_pr(re,y_label)


def vis_diff_roc(tag, in_origin, in_variant_arr, out_origin, out_variant_arr):
    """
    对比变化，origin预测结果和variant结果数组中，只要有任意不同，就找剔除
    :param tag: 输出TAG
    :param origin: 原标签
    :param variant_arr: 变体标签数组
    :return:
    """

    # 连续得分的结果，不变得分，变了不得分
    in_re = np.zeros(in_origin.shape[0])
    length = len(in_variant_arr)
    for variant in in_variant_arr:
        tfs = (np.argmax(in_origin, axis=1) == np.argmax(variant, axis=1))
        tfs = tfs / length
        in_re = in_re + tfs

    # print(in_re)

    # 连续得分的结果，不变得分，变了不得分
    out_re = np.zeros(out_origin.shape[0])
    length = len(out_variant_arr)
    for variant in out_variant_arr:
        tfs = (np.argmax(out_origin, axis=1) == np.argmax(variant, axis=1))
        tfs = tfs / length
        out_re = out_re + tfs

    # print(out_re)
    # print(in_re.shape)
    # print(out_re.shape)


    in_label = np.ones(in_re.shape)
    out_label = np.zeros(out_re.shape)

    y_label = np.hstack((in_label,out_label))
    re = np.hstack((in_re,out_re))
    # print(re.shape)
    # print(y_label.shape)
    cal_roc(re,y_label)



# 测试
if __name__ == "__main__":
    data = MNIST()
    cifar = CIFAR()
    #
    test_result = np.load('test_result.npy')
    compare_result('mnist', test_result)
    mnist_tf_results = []
    for i in range(10):
        result = np.load('exp_affine_in_%d.npy' % i)
        mnist_tf_results.append(result)
        compare_result('mnist %d' % i, result)
    print('\n')

    cifar_ood_results = []
    cifar_test_result = np.load('cifar_test_result.npy')
    for i in range(10):
        result = np.load('exp_affine_cifar_%d.npy' % i)
        cifar_ood_results.append(result)

    noise_ood_results = []
    noise_test_result = np.load('noise_result.npy')
    for i in range(10):
        result = np.load('exp_affine_noise_%d.npy' % i)
        noise_ood_results.append(result)

    compare_ood_result('cifar', cifar_test_result)
    for i in range(10):
        compare_ood_result('cifar ood %d' % i, cifar_ood_results[i])
        print('\n')
        compare_ood('cifar_cmp ood %d' % i, cifar_test_result, cifar_ood_results[i])
        print('\n')

    compare_ood_result('noise', noise_test_result)
    for i in range(10):
        compare_ood_result('noise ood %d' % i, noise_ood_results[i])
        print('\n')
        compare_ood('noise_cmp ood %d' % i, noise_test_result, noise_ood_results[i])
        print('\n')

    count_diff('mnist_in', test_result, mnist_tf_results)
    count_diff('cifar_ood', cifar_test_result, cifar_ood_results)
    count_diff('noise_ood', noise_test_result, noise_ood_results)

    vis_diff('vis cifar_ood', test_result, mnist_tf_results, cifar_test_result, cifar_ood_results)
    vis_diff('vis noise_ood', test_result, mnist_tf_results, noise_test_result, noise_ood_results)

    vis_diff_roc('vis roc cifar_ood', test_result, mnist_tf_results, cifar_test_result, cifar_ood_results)
    vis_diff_roc('vis roc noise_ood', test_result, mnist_tf_results, noise_test_result, noise_ood_results)


