# Parts of this script are taken from: https://github.com/bfortuner/pytorch_tiramisu.
#
# The source repository is under MIT License.
# Authors from original repository: Brendan Fortuner
#
# For more details and references checkout the repository and the readme of our repository.
#
# Author of this edited script: Anonymous

import matplotlib.pyplot as plt
import numpy as np

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

DSET_MEAN = [0.41189489566336, 0.4251328133025, 0.4326707089857]
DSET_STD = [0.27413549931506, 0.28506257482912, 0.28284674400252]

label_colours = np.array(
    [
        Sky,
        Building,
        Pole,
        Road,
        Pavement,
        Tree,
        SignSymbol,
        Fence,
        Car,
        Pedestrian,
        Bicyclist,
        Unlabelled,
    ]
)


def view_annotated(tensor, plot=True):
    temp = tensor.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 11):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0  # [:,:,0]
    rgb[:, :, 1] = g / 255.0  # [:,:,1]
    rgb[:, :, 2] = b / 255.0  # [:,:,2]
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def save_annotated(tensor, save_path):
    rgb = view_annotated(tensor, plot=False)
    plt.imsave(save_path, rgb)


def decode_image(tensor):
    inp = tensor.numpy().transpose((1, 2, 0))
    mean = np.array(DSET_MEAN)
    std = np.array(DSET_STD)
    inp = std * inp + mean
    return inp


def view_image(tensor):
    inp = decode_image(tensor)
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


def save_image(tensor, save_path):
    inp = decode_image(tensor)
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.savefig(save_path)
    plt.close()
