"""
Задание:
1. Реализовать генерацию ЦВЗ 𝛺 как псевдослучайной последовательности заданной длины из чисел,
распределённых по нормальному закону. Длина 1/2 плоскости из 3 пункта.

2. Реализовать трансформацию исходного контейнера к пространству признаков -
ДВП (вейвлеты Хаара), 3 уровня декомпозиции

3. Осуществить встраивание информации мультипликативным методом в плоскость HH спектра.

4. Сформировать носитель информации при помощи обратного преобразования от матрицы признаков к цифровому сигналу.
Сохранить его на диск

5. Считать носитель информации из файла и повторно выполнить п. 2 для носителя информации.

6. Сформировать оценку встроенного ЦВЗ 𝛺̃ неслепым методом (то есть с использованием матрицы
признаков исходного контейнера); выполнить детектирование при помощи функции близости 𝜌(𝛺, 𝛺̃)

7. Осуществить автоматический подбор значения параметра встраивания методом перебора с целью обеспечения
𝑃𝑆𝑁𝑅 > 30 дБ (или по желанию любого значения, большего 30 дБ), при этом выбирается
набор параметров, соответствующий наибольшему значению 𝜌.

8. Выполнить дополнительное исследование полученной системы встраивания информации - «Beta: Laplace»
"""

import numpy as np
import pywt
import skimage.metrics
import scipy
from skimage.io import imread, imshow, show, imsave
from matplotlib import pyplot as plt


def read_image(image_path):
    return np.array(imread(image_path)).astype(np.uint8)


# Прямое Вейвлет преобразование Хаара с заданным уровнем декомпозиции
def dwt(matrix: np.array, level: int):
    return pywt.wavedec2(matrix, 'haar', level=level)


# Обратное Вейвлет преобразование Хаара
def idwt(wavelet_coeffs: np.array):
    return pywt.waverec2(wavelet_coeffs, 'haar')


# Генерация нормально распределенного вектора с seed'ом
def generate_watermark(size: int, key: int, mean=0., spread=1.):
    rng = np.random.default_rng(key)
    return rng.normal(mean, spread, size)


# Мера близости
def get_rho(watermark: np.array, extracted_watermark: np.array):
    return np.cumsum(watermark * extracted_watermark)[-1] / \
           np.sqrt(np.cumsum(watermark ** 2)[-1] * np.cumsum(extracted_watermark ** 2)[-1])


def get_beta(matrix: np.array):
    mask = [[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]]  # лапласиан
    convolved = scipy.signal.convolve2d(matrix, mask, mode='same', boundary='symm', fillvalue=0)
    convolved = np.abs(convolved)  # избавление от отрицательных значений

    return convolved / np.max(convolved)


# Получение порога как медианное значение, сдвинутое на единицу влево
def get_threshold(matrix: np.array):
    sorted_matrix = np.sort(matrix, axis=None).reshape((matrix.shape[0], matrix.shape[1]))
    return sorted_matrix[(matrix.shape[0] - 1) // 2, (matrix.shape[1] - 1) - 1]


# Преобразование матрицы в массив диагоналей, полученных при зигзагообразном проходе матрицы
def get_matrix_diagonals(matrix: np.array):
    diagonals_count = matrix.shape[0] + matrix.shape[1] - 1

    horizontal_flip = np.fliplr(matrix)  # отражение вдоль ОУ, порядок сверху-вниз в исходной
    vertical_flip = np.flipud(matrix)  # отражение вдоль ОХ, порядок снизу-вверх в исходной

    diagonals = []

    # В каждом направлении по diagonals_count//2 диагоналей, обходим их все
    for i in range(-(diagonals_count // 2), diagonals_count // 2 + 1):
        if i % 2 == 0:  # нечетная диагональ (по номеру)
            diagonals.append(list(horizontal_flip.diagonal(-i)))
        else:  # четная диагональ (по номеру)
            diagonals.append(list(vertical_flip.diagonal(i)))

    return diagonals


# Преобразование массива диагоналей в матрицу
def diagonals_to_matrix(diagonals: np.array):
    diagonals_count = len(diagonals)

    matrix = np.zeros((diagonals_count // 2 + 1, diagonals_count // 2 + 1))

    horizontal_flip = np.fliplr(matrix)  # отражение вдоль ОУ, порядок сверху-вниз и справа-налево в исходной
    vertical_flip = np.flipud(matrix)  # отражение вдоль ОХ, порядок снизу-вверх и слева-направо в исходной

    # В каждом направлении по diagonals_count//2 диагоналей, обходим их все
    for i in range(-(diagonals_count // 2), diagonals_count // 2 + 1):
        if i % 2 == 0:  # нечетная диагональ (по номеру)
            if i > 0:
                np.fill_diagonal(horizontal_flip[i:, :],
                                 diagonals[i + diagonals_count // 2])  # компенсация для вставки (начиная с i = 1)
            else:  # отрицательное смещение относительно побочной диагонали (вставка в левую часть матрицы)
                np.fill_diagonal(horizontal_flip[:, -i:],
                                 diagonals[i + diagonals_count // 2])  # компенсация смещения
        else:  # четная диагональ (по номеру)
            if i > 0:
                np.fill_diagonal(vertical_flip[:, i:],
                                 diagonals[i + diagonals_count // 2])  # компенсация смещения
            else:  # отрицательное смещение относительно побочной диагонали (вставка в левую часть матрицы)
                np.fill_diagonal(vertical_flip[-i:, ],
                                 diagonals[i + diagonals_count // 2])  # компенсация смещения

    return matrix


def embedding(image: np.array, key: int, alpha: float, level: int, modified: bool):
    wavelet_coeffs = dwt(image, level)
    hh_zone = wavelet_coeffs[1][2]

    watermark_size = hh_zone.shape[0] * hh_zone.shape[1] // 2
    watermark = generate_watermark(watermark_size, key)

    diagonals = get_matrix_diagonals(hh_zone)

    threshold = get_threshold(hh_zone)
    w_i = 0  # для прохода по массиву watermark

    # Проход по элементам полученных диагоналей, начиная со 2-ой по номеру диагонали
    for i in range(1, len(diagonals)):  # обход всех диагоналей
        for j in range(len(diagonals[i])):  # обход элементов в диагонали
            if np.absolute(diagonals[i][j]) > np.absolute(threshold):
                diagonals[i][j] *= (1 + alpha * watermark[w_i])
                w_i += 1
                if w_i == watermark_size:
                    break
        else:
            continue  # выполнится в конце каждой итерации, если не было break из вложенного цикла
        break  # выполнится только в случае break вложенного цикла

    marked_hh_zone = diagonals_to_matrix(diagonals)

    # Из-за неизменяемости tuple приходится делать каст к list, менять в нем значения и затем вставлять обратно
    wavelet_coeffs_list = list(wavelet_coeffs[1])
    wavelet_coeffs_list[2] = marked_hh_zone
    wavelet_coeffs.pop(1)  # удаление исходного tuple
    wavelet_coeffs_tuple = tuple(wavelet_coeffs_list)
    wavelet_coeffs.insert(1, wavelet_coeffs_tuple)  # вставка измененного tuple

    if not modified:  # обычное встраивание
        return idwt(wavelet_coeffs), watermark
    else:  # модифицированное встраивание с параметром beta
        beta = get_beta(image)
        return idwt(wavelet_coeffs) * beta + image * (1 - beta), watermark


def extracting(marked_image: np.array, image: np.array, alpha: float, level: int):
    wavelet_coeffs = dwt(marked_image, level)
    marked_hh_zone = wavelet_coeffs[1][2]

    wavelet_coeffs = dwt(image, level)
    hh_zone = wavelet_coeffs[1][2]

    watermark_size = hh_zone.shape[0] * hh_zone.shape[1] // 2
    watermark = np.zeros(watermark_size)

    diagonals_original = get_matrix_diagonals(hh_zone)
    diagonals_marked = get_matrix_diagonals(marked_hh_zone)

    threshold = get_threshold(hh_zone)
    w_i = 0

    # Проход по элементам полученных диагоналей, начиная со 2-ой (по номеру) диагонали
    for i in range(1, len(diagonals_original)):  # обход всех диагоналей
        for j in range(len(diagonals_original[i])):  # обход элементов диагонали
            if np.absolute(diagonals_original[i][j]) > np.absolute(threshold):
                watermark[w_i] = (diagonals_marked[i][j] - diagonals_original[i][j]) / \
                                 (alpha * diagonals_original[i][j])
                w_i += 1
                if w_i == watermark_size:
                    return watermark

    return watermark


# Получение значения alpha, при котором rho максимально
def get_best_alpha(image: np.array, key: int, level: int, modified: bool):
    current_alpha = 0.1
    best_alpha = 0.
    max_rho = 0.

    for i in range(10):
        marked_image, watermark = embedding(image, key, current_alpha, level, modified=modified)
        extracted_watermark = extracting(marked_image, image, current_alpha, level)

        rho = get_rho(watermark, extracted_watermark)
        psnr = skimage.metrics.peak_signal_noise_ratio(image, marked_image)

        if rho > max_rho and psnr > 30:
            max_rho = rho
            best_alpha = current_alpha

        current_alpha += 0.1

    return best_alpha, max_rho


def watermark_detection(extracted_watermark, threshold, key):
    size = len(extracted_watermark)
    watermark = generate_watermark(size, key)
    rho = get_rho(watermark, extracted_watermark)
    if rho > threshold:
        return True
    else:
        return False


def plot(image, marked_image, watermark, extracted_watermark, alpha, rho, psnr, modified):
    print(f"Best alpha = {alpha}")
    print(f"Rho = {rho}")
    print(f"PSNR = {psnr}")
    print(f"MAE between watermarks = {np.average(np.abs(watermark - extracted_watermark))}")
    print(f"MAE between images = {np.average(np.abs(marked_image.astype(int) - image.astype(int)))}")

    fig = plt.figure()
    sp = fig.add_subplot(1, 2, 1)
    sp.set_title("Исходное изображение")
    imshow(image, cmap='gray')

    sp = fig.add_subplot(1, 2, 2)
    sp.set_title("Изображение с ЦВЗ")
    imshow(marked_image, cmap='gray', vmin=0, vmax=255)

    fig = plt.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.set_title("Difference between image and marked image")
    imshow(np.absolute(marked_image.astype(int) - image.astype(int)), cmap='gray')

    if modified:
        beta = get_beta(image)
        fig = plt.figure()
        sp = fig.add_subplot(1, 1, 1)
        sp.set_title("Beta")
        imshow(beta, cmap='gray')

    show()


def main():
    level = 3  # уровень декомпозиции
    image = read_image("./images/bridge.tif")
    key = 321
    modified = False
    detection_threshold = 0.2

    best_alpha, _ = get_best_alpha(image, key, level, modified)

    marked_image, watermark = embedding(image, key, best_alpha, level, modified)
    imsave("./images/marked_image.png", marked_image)

    marked_image = read_image("./images/marked_image.png")
    extracted_watermark = extracting(marked_image, image, best_alpha, level)

    rho = get_rho(watermark, extracted_watermark)
    psnr = skimage.metrics.peak_signal_noise_ratio(image, marked_image)

    print(f"Watermark with key {key} detected: {watermark_detection(extracted_watermark, detection_threshold, key)}")

    plot(image, marked_image, watermark, extracted_watermark, best_alpha, rho, psnr, modified)


main()
