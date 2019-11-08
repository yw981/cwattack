import tensorflow as tf
import numpy as np
from skimage import filters, transform
import matplotlib.pyplot as plt
import math

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel


def eval(label, arr):
    print(label + " count ", arr.shape[0], " max ", np.max(arr), " min ", np.min(arr), " mean ", np.mean(arr), " var ",
          np.var(arr), " median ", np.median(arr))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def compare_result(tag, result):
    softmax_scores = np.max(softmax(result), axis=1)
    amr = np.argmax(result, axis=1)
    aml = np.argmax(mnist.test_labels, axis=1)
    # print(amr)
    # print(aml)
    wrong_indices = (amr != aml)
    right_indices = ~wrong_indices
    print("acc = %f" % (1 - np.sum(wrong_indices) / aml.shape[0]))
    right_softmax_scores = softmax_scores[right_indices]
    wrong_softmax_scores = softmax_scores[wrong_indices]
    eval(tag + ' right', right_softmax_scores)
    eval(tag + ' wrong', wrong_softmax_scores)


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
            posp.imshow(img, cmap=plt.cm.gray)

    plt.show()


# 迁移到新的闭源工程！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
if __name__ == "__main__":
    with tf.Session() as sess:
        model = MNISTModel("models/mnist", sess)
        mnist = MNIST()
        np.random.seed(1234)
        k = 0
        params = []
        for i in range(30):
            print('it ' + str(i))

            tfparam = np.array([[1.1, 0., 0.], [0., 1.1, 0.], [0., 0., 1.]])
            tfseed = (np.random.rand(3, 3) - 0.5) * np.array([[0.2, 0.2, 6], [0.2, 0.2, 6], [0, 0, 0]])
            print(tfseed)
            tfparam += tfseed
            tform = transform.AffineTransform(tfparam)
            print(tfparam)

            # variant_mnist = [transform.warp(x + 0.5, tform) for x in mnist.test_data[:10, :, :, :]]
            variant_mnist = [transform.warp(x + 0.5, tform) for x in mnist.test_data]
            variant_mnist = np.reshape(variant_mnist, (10000, 28, 28, 1)) - 0.5

            mnist_test_result = model.model.predict(variant_mnist)
            amr = np.argmax(mnist_test_result, axis=1)
            aml = np.argmax(mnist.test_labels, axis=1)
            wrong_indices = (amr != aml)
            right_indices = ~wrong_indices
            acc = (1 - np.sum(wrong_indices) / aml.shape[0])
            print("acc = %f" % acc)
            if acc > 0.95:
                print('save #%d' % i)
                params.append(tfparam)
                np.save('exp_affine_in_%d.npy' % k, mnist_test_result)
                print(mnist_test_result.shape)
                k += 1
                if k >= 10:
                    break

        params = np.array(params)
        print(params.shape)
        np.save('tf_ood_params.npy', params)
        # eval('mnist_test_result', mnist_test_result)
        # compare_result('mnist', mnist_test_result)

        #
        # print(np.argmax(mnist_test_result[:10], axis=1))
        #
        # show_images(variant_mnist[:10, :, :, :])
