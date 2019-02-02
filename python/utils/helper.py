import matplotlib.pyplot as plt
import numpy as np
from mxnet.gluon.nn import Dense
from mxnet.ndarray import NDArray
import math


# def test_network(net, trainloader):
#
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
#
#     dataiter = iter(trainloader)
#     images, labels = dataiter.next()
#
#     # Create Variables for the inputs and targets
#     inputs = Variable(images)
#     targets = Variable(images)
#
#     # Clear the gradients from all Variables
#     optimizer.zero_grad()
#
#     # Forward pass, then backward pass, then update weights
#     output = net.forward(inputs)
#     loss = criterion(output, targets)
#     loss.backward()
#     optimizer.step()
#
#     return True


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    # image = image.numpy().transpose((1, 2, 0))
    #
    # if normalize:
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     image = std * image + mean
    #     image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def view_recon(img: NDArray, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.asnumpy().squeeze())
    axes[1].imshow(recon.asnumpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')


def view_classify(img: NDArray, ps: NDArray, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.asnumpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.asnumpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


def viewFCWeights(fc: Dense, inShape=(10, 10)):
    weight = fc.weight.data()

    viewWeights(weight, inShape)


def viewWeights(w: NDArray, inShape=(10, 10)):
    dOut = w.shape[0]
    dOutSqrt = int(math.sqrt(dOut))
    if dOutSqrt ** 2 < dOut:
        dOutSqrt += 1

    fig, axs = plt.subplots(dOutSqrt, dOutSqrt)
    ii = 0
    for x in range(0, dOutSqrt):
        for y in range(0, dOutSqrt):
            if ii < dOut:
                imshow(w[ii].reshape(inShape).asnumpy().squeeze(), ax=axs[x, y])
            ii += 1

    plt.tight_layout()
