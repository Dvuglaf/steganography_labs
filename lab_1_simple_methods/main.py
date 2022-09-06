# Задание (встраиваем в зеленый)
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


def get_layer(channel, idx):
    # 1. Занулить все биты, кроме нужной плоскости (&)
    # 2. Привести к бинарной матрице (сдвиг)
    return (channel & (2 ** (idx - 1))) >> (idx - 1)


def embedding_svi_1(image, watermark):
    copy_image = copy.copy(image)

    _, green, blue = get_channels(image)
    virtual_layer = get_layer(blue, 4) ^ get_layer(green, 1)  # "виртуальная" битовая плоскость

    embedded_layer = virtual_layer ^ watermark  # встраиваемая битовая плоскость

    new_green = (green & 254) | embedded_layer  # занулить первую битовую плоскость и встроить плоскость
    # new_green = (green & 254) | (embedded_layer ^ get_layer(green, 1))

    copy_image[:, :, 1] = new_green

    return copy_image


def extract_svi_2(image, marked_image):
    _, green, blue = get_channels(image)
    _, marked_green, _ = get_channels(marked_image)
    return get_layer(blue, 4) ^ get_layer(green, 1) ^ get_layer(marked_green, 1)


if __name__ == "__main__":
    image = read_image('./images/baboon.tif')
    copy_image = copy.copy(image)
    watermark = read_image('./images/mickey.tif') // 255  # (0, 255) -> (0, 1)

    marked_image = embedding_svi_1(image, watermark)
    extracted_watermark = extract_svi_2(image, marked_image)

    _, green, _ = get_channels(image)
    _, marked_green, _ = get_channels(marked_image)

    fig = plt.figure()
    sp = fig.add_subplot(2, 3, 1)
    sp.set_title("Исходное изображение")
    imshow(image)
    sp = fig.add_subplot(2, 3, 4)
    sp.set_title("Изображение с ЦВЗ")
    imshow(marked_image)
    sp = fig.add_subplot(2, 3, 2)
    sp.set_title("Исходный ЦВЗ")
    imshow(watermark, cmap='gray')
    sp = fig.add_subplot(2, 3, 5)
    sp.set_title("Извлеченный ЦВЗ")
    imshow(extracted_watermark, cmap='gray')
    sp = fig.add_subplot(2, 3, 3)
    sp.set_title("Зеленый канал исходного")
    imshow(green)
    sp = fig.add_subplot(2, 3, 6)
    sp.set_title("Зеленый канал с ЦВЗ")
    imshow(marked_green)

    show()

