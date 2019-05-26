import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from setup_mnist import MNIST, MNISTModel


# with tf.Session() as sess:
#     data, model = MNIST(), MNISTModel("models/mnist", sess)
#     # 因为原作定义的问题，必须model.model
#     advre = model.model.predict(data.test_data)
#     test_result = model.model.predict(data.test_data)
#     np.save('test_result', test_result)

def show_images(npdata, labels, error_preds, error_scores, num_row=10):
    fig = plt.figure()
    plt.subplots_adjust(wspace=1, hspace=1)
    rows = npdata.shape[0] // num_row + 1
    for i in range(npdata.shape[0]):
        img = npdata[i].reshape([28, 28])
        posp = fig.add_subplot(rows, num_row, i + 1)
        plt.title('%d : %d -> %d %.2f' % (i, labels[i], error_preds[i], error_scores[i]))
        posp.imshow(img, cmap=plt.cm.gray)

    plt.show()


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1,keepdims=True)


def decode(datum):
    return np.argmax(datum)

# nd = np.array([[3,1,0.2],[1,2,3]])
# print(softmax(nd))
data = MNIST()
test_result = np.load('test_result.npy')
print(test_result.shape)
# np.savetxt('softmax_results.npy',softmax(test_result))
softmax_scores = np.max(softmax(test_result), axis=1)
amr = np.argmax(test_result, axis=1)
aml = np.argmax(data.test_labels, axis=1)
print(amr)
print(aml)
error_indices = (amr != aml)
print("acc = %f" % (1 - np.sum(error_indices) / aml.shape[0]))
# print(np.sum(amr != aml))
error_pics = data.test_data[error_indices]
error_labels = np.argmax(data.test_labels[error_indices], axis=1)
error_preds = np.argmax(test_result[error_indices], axis=1)
error_scores = softmax_scores[error_indices]
print(error_scores.shape)
show_images(error_pics, error_labels, error_preds, error_scores, 8)
