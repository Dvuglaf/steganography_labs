# Задание:
# 1. Реализовать генерацию ЦВЗ 𝛺 как псевдослучайной последовательности заданной длины из чисел,
# распределённых по нормальному закону. Длина 1/2 плоскости из 3 пункта.
#
# 2. Реализовать трансформацию исходного контейнера к пространству признаков -
# ДВП (вейвлеты Хаара), 3 уровня декомпозиции
#
# 3. Осуществить встраивание информации мультипликативным методом в плоскость HH спектра.
#
# 4. Сформировать носитель информации при помощи обратного преобразования от матрицы признаков к цифровому сигналу.
# Сохранить его на диск
#
# 5. Считать носитель информации из файла и повторно выполнить п. 2 для носителя информации.
#
# 6. Сформировать оценку встроенного ЦВЗ 𝛺̃ неслепым методом (то есть с использованием матрицы
# признаков исходного контейнера); выполнить детектирование при помощи функции близости 𝜌(𝛺,𝛺̃)
#
# 7. Осуществить автоматический подбор значения параметра встраивания методом перебора с целью обеспечения
# 𝑃𝑆𝑁𝑅 > 30 дБ (или по желанию любого значения, большего 30 дБ), при этом выбирается
# набор параметров, соответствующий наибольшему значению 𝜌.
#
# 8. Выполнить дополнительное исследование полученной системы встраивания информации - «Beta: Laplace»
import numpy as np
import pywt
from skimage.io import imread, imshow, show
from matplotlib import pyplot as plt

def read_image(image_path):
    return np.array(imread(image_path)).astype(np.int32)


def zigzag(original: np.ndarray, watermark: np.ndarray, to_embed: bool, embedded: np.ndarray, alpha=1.):
    sorted_matrix = np.sort(original, axis=None).reshape((original.shape[0], original.shape[1]))
    threshold = sorted_matrix[(original.shape[0] - 1) // 2, (original.shape[1] - 1) - 1]
    diag_num = original.shape[0] + original.shape[1] - 1
    i = 0
    j = 1
    m = 0
    n = 1
    for k in range(1, diag_num):  # номер диагонали
        if i > original.shape[0] - 1:  # граничные условия по строке
            i -= 1
            j += 1
        if j > original.shape[1] - 1:  # граничные условия по столбцу
            j -= 1
            i += 1
        if k > diag_num // 2:  # граничные условия по количеству элементов в диагонали ниже побочной диагонали
            n -= 1
        else:
            n += 1
        for l in range(n):  # номер элемента в диагонали
            if np.absolute(original[i][j]) > np.absolute(threshold):
                if to_embed:
                    original[i][j] *= (1 + alpha * watermark[m])
                else:
                    watermark[m] = (embedded[i][j] - original[i][j]) / (alpha * original[i][j])
                m += 1
                if m > len(watermark) - 1:
                    return None
            if k % 2 == 0:  # четная диагональ
                i -= 1
                j += 1
            else:  # нечетная диагональ
                j -= 1
                i += 1

        if k % 2 == 0:
            i += 1
        else:
            j += 1

    return None

def embedding(image: np.ndarray, key: int, alpha: float, level: int, mean=0., spread=1.):
    wavelet_coeffs = dwt(image, level)
    hh_zone = wavelet_coeffs[1][2]

    size = hh_zone.shape[0] * hh_zone.shape[1] // 2
    watermark = generate_watermark(size, key, mean, spread)


    zigzag(hh_zone, watermark, True, np.zeros(1), alpha)

    wavelet_coeffs_list = list(wavelet_coeffs[1])
    wavelet_coeffs_list[2] = hh_zone
    wavelet_coeffs.pop(1)
    wavelet_coeffs_tuple = tuple(wavelet_coeffs_list)
    wavelet_coeffs.insert(1, wavelet_coeffs_tuple)
    return idwt(wavelet_coeffs), watermark


def extracting(embedded_image: np.ndarray, image: np.ndarray, level: int):
    wavelet_coeffs = dwt(embedded_image, level)
    hh_zone_embedded = wavelet_coeffs[1][2]
    wavelet_coeffs = dwt(image, level)
    hh_zone = wavelet_coeffs[1][2]

    size = hh_zone.shape[0] * hh_zone.shape[1] // 2
    watermark = np.zeros(size)

    zigzag(hh_zone, watermark, False, hh_zone_embedded)

    return watermark

def dwt(image: np.ndarray, level: int):
    return pywt.wavedec2(image, 'haar', level=level)


def idwt(wavelet_coeffs):
    return pywt.waverec2(wavelet_coeffs, 'haar')

def generate_watermark(size: int, key: int, mean: float, spread: float):
    rng = np.random.default_rng(key)
    return rng.normal(mean, spread, size)

def main():
    level = 3  # уровень декомпозиции
    mean = 0.  # МО
    spread = 1.  # СКО
    image = imread("./images/bridge.tif")
    key = 321
    alpha = 1.

    embedded_image, watermark = embedding(image, key, alpha, level)
    extracted_watermark = extracting(embedded_image, image, level)

    print(np.max(np.abs(extracted_watermark - watermark)))

    plt.figure()

    imshow(embedded_image - image, cmap='gray')

    plt.figure()

    imshow(image, cmap='gray')

    show()


main()