import tensorflow as tf
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters, transform

from setup_cifar import CIFAR, CIFARModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi
import math


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
#         # data, model =  MNIST(), MNISTModel("models/mnist", sess)
#         data, model =  CIFAR(), CIFARModel("models/cifar", sess)
#         # attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
#         attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,largest_const=15)
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

def show_images(npdata, num_row=10):
    fig = plt.figure()
    plt.subplots_adjust(wspace=1)
    rows = npdata.shape[0] // num_row + 1  # 加一行对比
    for i in range(npdata.shape[0]):
        img = npdata[i] + 0.5
        posp = fig.add_subplot(rows, num_row, i + 1)
        # plt.title('%d : %d -> %d %.2f' % (i, labels[i], error_preds[i], error_scores[i]))
        posp.imshow(img)

    for i in range(num_row):
        dif = npdata[i + num_row] - npdata[i]
        posp = fig.add_subplot(rows, num_row, num_row * 2 + i + 1)
        posp.imshow(dif)

    plt.show()


# 原版
if __name__ == "__main__":
    with tf.Session() as sess:
        data, model = CIFAR(), CIFARModel("models/cifar", sess)
        #     # attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
        #     attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,largest_const=15)
        #
        #     inputs, targets = generate_data(data, samples=1, targeted=True,
        #                                     start=0, inception=False)
        #     timestart = time.time()
        #     adv_cifar = attack.attack(inputs, targets)
        #     timeend = time.time()
        #
        #     print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
        #
        #     np.save('adv_cifar',adv_cifar)
        adv = np.load('adv_cifar.npy')
        print(adv.shape)

        np.random.seed(1234)
        # #=========测试攻击效果
        # 取正常图像做实验，通过噪声、高斯、平移、旋转、缩放、剪切
        # adv = data.test_data[:9,:,:]
        # 取OOD，纯噪声图像
        # adv = np.random.randn(9, 28, 28, 1)

        # #=========给生成的对抗样本加一些随机噪声

        # noise = np.random.rand(adv.shape[0], adv.shape[1], adv.shape[2], adv.shape[3])
        # # adv样本0.8 0.996
        # noise[noise < 0.996] = 0
        # nadv = adv + noise
        # #=======增加噪声结束，若用噪声图像，请用nadv

        # #=========给生成的对抗样本加高斯过滤
        # sigma = 0.5
        # nadv = []
        # for i in range(adv.shape[0]):
        #     ge = filters.gaussian(adv[i], sigma)
        #     nadv.append(ge)
        # nadv = np.array(nadv)
        # #=======高斯过滤结束，请用nadv

        # #=========对抗样本AffineTransform
        nadv = []
        # tform = transform.AffineTransform(translation=(1, 0)) # 平移（x,y) x正水平向右，y正竖直向上
        # tform = transform.AffineTransform(scale=(0.98, 0.98))  # scale : (sx, sy) 缩放，x,y的比例
        tform = transform.AffineTransform(rotation=-3.14/24) # 逆时针旋转，弧度，绕左上角顶点
        # tform = transform.AffineTransform(shear=3.14/24)
        for i in range(adv.shape[0]):
            ge = transform.warp((adv[i] + 0.5), tform)
            nadv.append(ge - 0.5)
        nadv = np.array(nadv)
        # #=======AffineTransform结束，请用nadv

        advre = model.model.predict(adv)
        # 0-airplane,1-automobile,2-bird,3-cat,4-deer,5-dog,6-frog,7-horse,8-ship,9-truck
        print(np.argmax(advre, axis=1))
        # show_images(adv_cifar)

        nadvre = model.model.predict(nadv)
        print(nadvre.shape)
        print(np.argmax(nadvre, axis=1))
        show_images(np.concatenate((adv, nadv), axis=0), 9)
