## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel


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


def show_images(images):
    width = height = int(math.ceil(math.sqrt(images.shape[0])))
    fig = plt.figure()
    plt.subplots_adjust(wspace=1, hspace=1)
    for i in range(height):
        for j in range(width):
            idx = (i * width) + j
            if idx >= images.shape[0]:
                plt.show()
                return
            img = images[idx].reshape([28, 28])
            posp = fig.add_subplot(height, width, (i * width) + j + 1)
            posp.imshow(img)

    plt.show()


def generate_attack(from_image, to_image, epsilon=16 / 256):
    dis = to_image - from_image
    generate_image = from_image
    ops = np.fabs(dis) < epsilon
    generate_image[ops] = to_image[ops]
    ops = np.fabs(dis) >= epsilon
    generate_image[ops] += np.sign(to_image[ops]) * epsilon
    return generate_image


# 测试
if __name__ == "__main__":
    with tf.Session() as sess:
        data, model = MNIST(), MNISTModel("models/mnist", sess)
        sample_tests = data.test_data[0:100] + 0.5
        sample_tos = data.test_data[1:101] + 0.5
        adv = generate_attack(sample_tests, sample_tos) - 0.5

        advre = model.model.predict(adv)
        print(np.argmax(advre, axis=1))
        show_images(adv)
        # nadvre = model.model.predict(nadv)
        # print(nadvre.shape)
        # print(np.argmax(nadvre, axis=1))
        # show_images(np.concatenate((adv, nadv), axis=0), 9)
        # # np.save('advre', advre)
