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

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


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


# 原版
# if __name__ == "__main__":
#     with tf.Session() as sess:
#         data, model =  MNIST(), MNISTModel("models/mnist", sess)
#         # data, model =  CIFAR(), CIFARModel("models/cifar", sess)
#         attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
#         # attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,largest_const=15)
#
#         inputs, targets = generate_data(data, samples=1, targeted=True,
#                                         start=0, inception=False)
#         timestart = time.time()
#         adv = attack.attack(inputs, targets)
#         timeend = time.time()
#
#         print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
#
#         for i in range(len(adv)):
#             print("Valid:")
#             show(inputs[i])
#             print("Adversarial:")
#             show(adv[i])
#
#             print("Classification:", model.model.predict(adv[i:i+1]))
#
#             print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)

# 测试
if __name__ == "__main__":
    # with tf.Session() as sess:
    #     data, model = MNIST(), MNISTModel("models/mnist", sess)
    #     # inputs, targets = generate_data(data, samples=1, targeted=True,start=0, inception=False)
    #     # print(data.test_data.shape)
    #     # print(inputs.shape)
    #     # print(targets.shape)
    #     # print(targets)
    #     attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
    #     inputs, targets = generate_data(data, samples=1, targeted=True,
    #                                     start=0, inception=False)
    #     timestart = time.time()
    #     # attack返回的adv是np.array
    #     adv = attack.attack(inputs, targets)
    #     timeend = time.time()
    #
    #     print("Took", timeend - timestart, "seconds to run", len(inputs), "samples.")
    #
    #     np.save('adv', adv)
    #     print(adv.shape)

    adv = np.load('adv.npy')
    print(adv.shape[0])
    for i in range(adv.shape[0]):
        print(adv[i].shape)
        # 保存成图片
        Image.fromarray((adv[i].reshape((adv[i].shape[0], adv[i].shape[1])) + 0.5) * 255)\
            .convert('1').save('adv_'+str(i)+'.bmp')

