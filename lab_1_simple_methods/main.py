# Лабораторная работа 1.
# Вариант 24
# Номера НЗБП в СВИ-1 (𝑝) с указанием цветовых каналов: Blue-4 XOR Green-1 (встраиваем в зеленый: 232 vs 150)
# Цветовой канал, используемый в СВИ-4: Cb
# Способ встраивания в СВИ-4: (3.13), add = С (mod delta)

import bitarray
import copy
import numpy as np
from skimage.io import imread, imshow, show
from matplotlib import pyplot as plt

VAR = 24
DELTA = 4 + 4 * (VAR % 3)
IMAGE_SHAPE = (512, 512, 3)
WATERMARK_SHAPE = (512, 512)


def read_image(image_path):
    return np.array(imread(image_path)).astype(np.int32)


def get_channels(image):
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]


# Получение битовой плоскости канала
def get_bit_layer(channel, idx: int):
    # 1. Занулить все биты, кроме нужной плоскости (&)
    # 2. Привести к бинарной матрице (сдвиг)
    return (channel & (2 ** (idx - 1))) >> (idx - 1)


# Генерация с seed'ом заданного количества уникальных координат
def generate_coordinates(seed, count):
    rng = np.random.default_rng(seed)  # создание нового битового генератора с seed'ом

    # Генерация массива без повторений и получение координат (остаток от деления, целая часть)
    unique_numbers = rng.choice(IMAGE_SHAPE[0] * IMAGE_SHAPE[1], size=count, replace=False)
    return np.array((unique_numbers % IMAGE_SHAPE[0], unique_numbers // IMAGE_SHAPE[0])).T


# Преобразование битового вектора в матрицу со случайными координатами
def bit_vector_to_watermark(bit_vector, seed):
    coordinates = generate_coordinates(seed, len(bit_vector))

    # Преобразование битового вектора в матрицу на основе сгенерированных координат
    watermark = np.zeros(WATERMARK_SHAPE, dtype=np.uint8)

    for i in range(len(coordinates)):
        watermark[coordinates[i][0], coordinates[i][1]] = bit_vector[i]

    return watermark


# НЗБ-встраивание
def embedding_svi_1(image, watermark):
    copy_image = copy.copy(image)

    _, green, blue = get_channels(image)
    virtual_layer = get_bit_layer(blue, 4) ^ get_bit_layer(green, 1)  # "виртуальная" битовая плоскость

    embedding_layer = virtual_layer ^ watermark  # встраиваемая битовая плоскость (3.4)

    new_green = (green & 254) | embedding_layer  # занулить первую битовую плоскость и встроить embedded_layer
    # new_green = (green & 254) | (embedded_layer ^ get_layer(green, 1))

    copy_image[:, :, 1] = new_green

    return copy_image


def extracting_svi_1(image, marked_image):
    _, green, blue = get_channels(image)
    _, marked_green, _ = get_channels(marked_image)
    return get_bit_layer(blue, 4) ^ get_bit_layer(green, 1) ^ get_bit_layer(marked_green, 1)


# Стеганографическое НЗБ-встраивание: key - длина встраиваемой строки, key * 8 - значение seed'а для генератора
def embedding_svi_2(image, text: str, key: int):
    bit_vector = bitarray.bitarray()
    bit_vector.frombytes(text.encode('utf-8'))

    q = len(bit_vector) / (IMAGE_SHAPE[0] * IMAGE_SHAPE[1])  # заполненность контейнера
    print(f"\tq: {q}")

    if q > 1:
        raise ValueError("Container overflow, string is too big!")

    coordinates = generate_coordinates(key, len(bit_vector))

    copy_image = copy.copy(image)

    _, green, _ = get_channels(copy_image)

    embedding_layer = get_bit_layer(green, 1)

    # Встраивание в сгенерированные координаты
    for i in range(len(coordinates)):
        embedding_layer[coordinates[i][0], coordinates[i][1]] = bit_vector[i]

    new_green = (green & 254) | embedding_layer  # занулить первую битовую плоскость и встроить embedded_layer

    copy_image[:, :, 1] = new_green
    return copy_image


def extracting_svi_2(marked_image, key: int):
    _, marked_green, _ = get_channels(marked_image)
    marked_layer = get_bit_layer(marked_green, 1)

    rng = np.random.default_rng(key)  # создание нового битового генератора с seed'ом
    extracted_bit_vector = bitarray.bitarray(key * 8)

    # Генерация массива без повторений и получение координат (остаток от деления, целая часть)
    unique_numbers = rng.choice(IMAGE_SHAPE[0] * IMAGE_SHAPE[1], size=key * 8, replace=False)
    coordinates = np.array((unique_numbers % IMAGE_SHAPE[0], unique_numbers // IMAGE_SHAPE[0])).T

    # Извлечение из битовой плоскости в сгенерированных координатах
    for i in range(key * 8):
        extracted_bit_vector[i] = marked_layer[coordinates[i][0], coordinates[i][1]]

    return extracted_bit_vector.tobytes().decode('utf-8')  # перевод в строку


# Стеганографическое НЗБ-встраивание с данными из задания 1-2: key - длина встраиваемой строки,
#                                                              key * 8 - значение seed'а для генератора
def embedding_svi_2_virtual(image, text: str, key: int):
    bit_vector = bitarray.bitarray()
    bit_vector.frombytes(text.encode('utf-8'))

    q = len(bit_vector) / (IMAGE_SHAPE[0] * IMAGE_SHAPE[1])  # заполненность контейнера
    print(f"\tq: {q}")

    if q > 1:
        raise ValueError("Container overflow, string is too big!")

    watermark = bit_vector_to_watermark(bit_vector, key)
    return embedding_svi_1(image, watermark), watermark


def extracting_svi_2_virtual(image, marked_image, key: int):
    extracted_watermark = extracting_svi_1(image, marked_image)

    extracted_bit_vector = bitarray.bitarray(key * 8)

    coordinates = generate_coordinates(key, key * 8)

    # Извлечение
    for i in range(key * 8):
        extracted_bit_vector[i] = extracted_watermark[coordinates[i][0], coordinates[i][1]]

    return extracted_bit_vector.tobytes().decode('utf-8'), bit_vector_to_watermark(extracted_bit_vector, key)


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


def plot(title, image, marked_image, watermark, extracted_watermark, channel, marked_channel, channel_name):
    fig = plt.figure()
    fig.suptitle(title)

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

    print(f"\tМаксимальное значение ошибки в канале: {np.max(np.abs(marked_channel - channel))}\n")


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


# СВИ-1
def task_1(image, watermark):
    print("Задание 1-2 (СВИ-1):")

    marked_image = embedding_svi_1(image, watermark)
    extracted_watermark = extracting_svi_1(image, marked_image)

    _, green, _ = get_channels(image)
    _, marked_green, _ = get_channels(marked_image)

    plot("Задание 1-2 (СВИ-1)", image, marked_image, watermark, extracted_watermark, green, marked_green, "Зеленый")


# СВИ-4
def task_2(image, watermark):
    print("Задание 3-4 (СВИ-4):")

    converted_image = rgb2ycbcr(image)

    marked_image = embedding_svi_4(converted_image, watermark)
    extracted_watermark = extracting_svi_4(converted_image, marked_image)

    _, cb, _ = get_channels(converted_image)
    _, marked_cb, _ = get_channels(marked_image)

    marked_image = ycbcr2rgb(marked_image).astype(np.int32)
    plot("Задание 3-4 (СВИ-4)", image, marked_image, watermark, extracted_watermark, cb, marked_cb, "Cb")


# СВИ-1 и СВИ-4
def extra_task_3(image, watermark_1, watermark_2):
    print("Допзадание 1_1:")

    marked_image = embedding_svi_1(image, watermark_1)

    converted_image = rgb2ycbcr(marked_image)
    marked_image = embedding_svi_4(converted_image, watermark_2)

    _, cb, _ = get_channels(converted_image)
    _, marked_cb, _ = get_channels(marked_image)

    extracted_watermark_2 = extracting_svi_4(converted_image, marked_image)

    marked_image = ycbcr2rgb(marked_image).astype(np.int32)
    extracted_watermark = extracting_svi_1(image, marked_image)

    _, green, _ = get_channels(image)
    _, marked_green, _ = get_channels(marked_image)

    plot("Допзадание 1_1", image, marked_image, watermark_1, extracted_watermark, green, marked_green, "Зеленый")

    print("Допзадание 1_2:")
    plot("Допзадание 1_2", image, marked_image, watermark_2, extracted_watermark_2, cb, marked_cb, "Cb")


# СВИ-2: key - длина встраиваемой строки, key * 8 - значение seed'а для генератора
def extra_task_4(image, text):
    print("Допзадание 2 (СВИ-2):")

    marked_image = embedding_svi_2(image, text, len(text))
    extracted_text = extracting_svi_2(marked_image, len(text))

    print(f"\tOriginal:\t{text}\n\tExtracted:\t{extracted_text}")

    _, green, _ = get_channels(image)
    _, marked_green, _ = get_channels(marked_image)

    plot("Допзадание 2 (СВИ-2)", image, marked_image, np.zeros(WATERMARK_SHAPE),
         np.zeros(WATERMARK_SHAPE), green, marked_green, "Зеленый")


# СВИ-2 на основании данных из задания 1-2: key - длина встраиваемой строки, key * 8 - значение seed'а для генератора
def extra_task_4_virtual(image, text):
    print("Допзадание 2 (СВИ-2):")

    marked_image, watermark = embedding_svi_2_virtual(image, text, len(text))
    extracted_text, extracted_watermark = extracting_svi_2_virtual(image, marked_image, len(text))

    print(f"\tOriginal:\t{text}\n\tExtracted:\t{extracted_text}")

    _, green, _ = get_channels(image)
    _, marked_green, _ = get_channels(marked_image)

    plot("Допзадание 2 (СВИ-2)", image, marked_image, watermark, extracted_watermark, green, marked_green, "Зеленый")


def main():
    image = read_image('./images/baboon.tif')

    watermark_1 = read_image('./images/mickey.tif') // 255
    watermark_2 = read_image('./images/ornament.tif') // 255

    text = "Test string for SVI-2 method!"

    task_1(image, watermark_1)
    task_2(image, watermark_1)
    extra_task_3(image, watermark_1, watermark_2)
    extra_task_4_virtual(image, text)

    show()


main()
