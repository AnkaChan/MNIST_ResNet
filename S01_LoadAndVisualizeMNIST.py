
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import itertools

def visualizeImgsAndLabel(imgs, labels, gridW=10, gridH=10, shuffle=True, cmap='gray', figTitle=None, outFile=None, outDPI=300, closeExistingFigures=False):
    if closeExistingFigures:
        matplotlib.plt.close("all")
    fig, axs = plt.subplots(gridH, gridW)
    fig.set_size_inches(20*gridW/10, 20*gridH/10)
    padSize = 100

    indices = list(range((len(imgs))))
    if shuffle:
        np.random.shuffle(indices)

    for i, j in itertools.product(range(gridH), range(gridW)):
        imgId = indices[i * gridW + j]
        img = imgs[imgId, ...]
        # pProj = testgtset[i * gridW + j, :]
        axs[i, j].imshow(np.squeeze(img), cmap=cmap)
        axs[i, j].set_title( labels[imgId])
        axs[i, j].axis('off')

    if figTitle is not None:
        fig.suptitle(figTitle)

    if outFile is not None:
        fig.savefig(outFile, dpi=outDPI, transparent=True, bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':

    imgsTrainFlatten = mnist.train.images
    labelTrain = mnist.train.labels

    imgsTestOrgFlatten = mnist.test.images
    labelTest = mnist.test.labels

    imgsTrain = imgsTrainFlatten.reshape(-1, 28, 28, 1)
    imgsTest = imgsTestOrgFlatten.reshape(-1, 28, 28, 1)


    def labelsToText(labels, logitsInput=True):
        if logitsInput:
            labelsId = np.argmax(labels, axis=1)
        else:
            labelsId = labels
        labelText = [str(l) for l in labelsId]

        return labelText


    # %%

    # Visualize the data
    labelTrainText = labelsToText(labelTrain)
    labelTestText = labelsToText(labelTest)

    visualizeImgsAndLabel(imgsTrain, labelTrainText, figTitle="TrainSet")
    visualizeImgsAndLabel(imgsTest, labelTestText, figTitle="TestSet")

    plt.show()
    plt.waitforbuttonpress()
