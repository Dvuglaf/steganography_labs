# Лабораторная работа 1.
# Вариант 24
# Номера НЗБП в СВИ-1 (𝑝) с указанием цветовых каналов: Blue-4 XOR Green-1 (встраиваем в зеленый: 232 vs 150)
# Цветовой канал, используемый в СВИ-4: Cb
# Способ встраивания в СВИ-4: (3.13), add = С (mod delta)

import copy
import numpy as np
from skimage.io import imread, imshow, show
from matplotlib import pyplot as plt


VAR = 24
DELTA = (4 + 4 * VAR) % 3
IMAGE_SHAPE = (512, 512, 3)


def read_image(image_path):
    return np.array(imread(image_path)).astype(np.int32)


def get_channels(image):
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]


def get_bit_layer(channel, idx):
    # 1. Занулить все биты, кроме нужной плоскости (&)
    # 2. Привести к бинарной матрице (сдвиг)
    return (channel & (2 ** (idx - 1))) >> (idx - 1)


# НЗБ-встраивание
def embedding_svi_1(image, watermark):
    copy_image = copy.copy(image)

    _, green, blue = get_channels(image)
    virtual_layer = get_bit_layer(blue, 4) ^ get_bit_layer(green, 1)  # "виртуальная" битовая плоскость

    embedded_layer = virtual_layer ^ watermark  # встраиваемая битовая плоскость (3.4)

    new_green = (green & 254) | embedded_layer  # занулить первую битовую плоскость и встроить embedded_layer
    # new_green = (green & 254) | (embedded_layer ^ get_layer(green, 1))

    copy_image[:, :, 1] = new_green

    return copy_image


def extract_svi_1(image, marked_image):
    _, green, blue = get_channels(image)
    _, marked_green, _ = get_channels(marked_image)
    return get_bit_layer(blue, 4) ^ get_bit_layer(green, 1) ^ get_bit_layer(marked_green, 1)


# Simple-QIM метод встраивания
def embedding_svi_4(image, watermark):
    copy_image = copy.copy(image)

    additive = image[:, :, 1] % DELTA  # (3.13)
    new_cb = (image // (2 * DELTA) * (2 * DELTA))[:, :, 1] + watermark * DELTA + additive  # (3.10)
    copy_image[:, :, 1] = new_cb
    return copy_image


def extracting_svi_4(image, marked_image):
    additive = image[:, :, 1] % DELTA
    return (marked_image[:, :, 1] - additive - (image // (2 * DELTA) * (2 * DELTA))[:, :, 1]) / DELTA


def plot(image, marked_image, watermark, extracted_watermark, channel, marked_channel, channel_name):
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
    sp.set_title(f"{channel_name} канал исходного")
    imshow(channel)

    sp = fig.add_subplot(2, 3, 6)
    sp.set_title(f"{channel_name} канал с ЦВЗ")
    imshow(marked_channel)


# Перевод RGB -> YCbCr
def rgb2ycbcr(image):
    y = (77 / 256) * image[:, :, 0] + (150 / 256) * image[:, :, 1] + (29 / 256) * image[:, :, 2]
    cb = image[:, :, 2] - y
    cr = image[:, :, 0] - y

    ycbcr_image = np.zeros(IMAGE_SHAPE)
    ycbcr_image[:, :, 0] = y
    ycbcr_image[:, :, 1] = cb
    ycbcr_image[:, :, 2] = cr

    return ycbcr_image


# Перевод YCbCr -> RGB
def ycbcr2rgb(image):
    red = image[:, :, 2] + image[:, :, 0]
    blue = image[:, :, 1] + image[:, :, 0]
    green = (image[:, :, 0] - (77 / 256) * red - (29 / 256) * blue) * (256 / 150)

    rgb_image = np.zeros(IMAGE_SHAPE)
    rgb_image[:, :, 0] = red
    rgb_image[:, :, 1] = green
    rgb_image[:, :, 2] = blue

    return rgb_image


def main():
    image = read_image('./images/baboon.tif')
    watermark = read_image('./images/mickey.tif') // 255  # (0, 255) -> (0, 1)

    # СВИ-1
    marked_image = embedding_svi_1(image, watermark)
    extracted_watermark = extract_svi_1(image, marked_image)

    _, green, _ = get_channels(image)
    _, marked_green, _ = get_channels(marked_image)

    plot(image, marked_image, watermark, extracted_watermark, green, marked_green, "Зеленый")

    # СВИ-4
    converted_image = rgb2ycbcr(image)

    marked_image = embedding_svi_4(converted_image, watermark)
    extracted_watermark = extracting_svi_4(converted_image, marked_image)

    _, cb, _ = get_channels(converted_image)
    _, marked_cb, _ = get_channels(marked_image)

    converted_image = ycbcr2rgb(converted_image).astype(np.int32)
    marked_image = ycbcr2rgb(marked_image).astype(np.int32)
    plot(converted_image, marked_image, watermark, extracted_watermark, cb, marked_cb, "Cb")

    show()


main()
