# Задание на лабу (встраиваем в зеленый)
# Номера НЗБП в СВИ-1 (𝑝) с указанием цветовых каналов:
# Blue-4 XOR Green-1
# Цветовой канал, используемый в СВИ-4:
# Cb
import copy

from skimage.io import imread, imshow, show
from matplotlib import pyplot as plt
import numpy as np


def read_image(image_path):
    return np.array(imread(image_path)).astype(np.int32)


def get_channels(image):
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]


def get_plane(channel, idx):
    return (channel & (2 ** (idx - 1))) >> (idx - 1)


if __name__ == "__main__":
    image = read_image('./images/baboon.tif')
    copy_image = copy.copy(image)
    watermark = read_image('./images/mickey.tif') // 255

    _, green, blue = get_channels(image)
    plane = get_plane(blue, 4) ^ get_plane(green, 1)
    temp = plane ^ watermark
    new_green = (green & 254) | temp
    copy_image[:, :, 1] = new_green

    _, copy_green, _ = get_channels(copy_image)

    extracted = get_plane(blue, 4) ^ get_plane(green, 1) ^ get_plane(copy_green, 1)

    fig = plt.figure()
    sub = fig.add_subplot(1, 2, 1)
    imshow(watermark)
    sub = fig.add_subplot(1, 2, 2)
    imshow(extracted)
    show()

