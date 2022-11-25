"""
Task:
CyclicShift, Scale, Median, JPEG, Beta: Laplace

"""
import numpy as np
from skimage.io import imread, imshow, show, imsave
from skimage.transform import rescale
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt
from lab_2_spectrum_expansion.main import read_image, embedding, extracting, get_rho


def cyclic_shift(marked_image, r):
    N_1, N_2 = marked_image.shape
    shifted = np.roll(marked_image, int(r * N_1), axis=1)
    shifted = np.roll(shifted, int(r * N_2), axis=0)
    return shifted

"""
Циклический сдвиг
Результат:
ЦВЗ не извлекается, мера близости около 0.02
ДВП (Хаара) не устойчиво к геометрическому преобразованию "Циклический сдвиг",
поскольку оно основано на контурах изображения,
которые, в свою очередь, смещаются при выполнении преобразования,
что делает невозможным корректное извлечение ЦВЗ
"""
def task1(image, marked_image, watermark, alpha, level):
    rho_shifted_arr = []

    for r in np.arange(0.1, 1, 0.1):
        shifted_image = cyclic_shift(marked_image, r)

        extracted_watermark = extracting(marked_image, image, alpha, level)
        extracted_watermark_shifted = extracting(shifted_image, image, alpha, level)

        rho = get_rho(watermark, extracted_watermark)
        rho_shifted = get_rho(watermark, extracted_watermark_shifted)

        rho_shifted_arr.append(rho_shifted if rho_shifted > 0 else 0)

        if r == 0.1:
            print(f"Rho: {rho}")

        if r == 0.4:
            fig = plt.figure()
            plt.suptitle(f"r={r}")
            sp = fig.add_subplot(1, 2, 1)
            sp.set_title("Исходное изображение")
            imshow(marked_image, cmap='gray', vmin=0, vmax=255)

            sp = fig.add_subplot(1, 2, 2)
            sp.set_title("Сдвинутое изображение")
            imshow(shifted_image, cmap='gray', vmin=0, vmax=255)

    plt.figure()
    plt.title("Зависимость меры близости от r")
    plt.plot(np.arange(0.1, 1, 0.1), rho_shifted_arr)


def scale(marked_image, k):
    N_1, N_2 = marked_image.shape
    scaled_image = rescale(marked_image, k)
    N_1_scaled, N_2_scaled = scaled_image.shape
    if k < 1.:
        result_image = np.zeros(marked_image.shape)
        result_image[N_1 // 2 - N_1_scaled // 2: N_1 // 2 + N_1_scaled // 2,
                     N_2 // 2 - N_2_scaled // 2: N_2 // 2 + N_2_scaled // 2] =\
            scaled_image[N_1_scaled // 2 - N_1_scaled // 2: N_1_scaled // 2 + N_1_scaled // 2,
                         N_2_scaled // 2 - N_2_scaled // 2: N_2_scaled // 2 + N_2_scaled // 2]
        return result_image
    elif k > 1.:
        return scaled_image[N_1_scaled // 2 - N_1 // 2: N_1_scaled // 2 + N_1 // 2,
                            N_2_scaled // 2 - N_2 // 2: N_2_scaled // 2 + N_2 // 2]
    else:
        return marked_image


"""
Масштабирование
Результат:
ЦВЗ не извлекается, мера близости около 0.006
ДВП (Хаара) не устойчиво к геометрическому преобразованию "Масштабирование",
поскольку оно основано на контурах изображения,
которые, в свою очередь, смещаются при выполнении преобразования,
что делает невозможным корректное извлечение ЦВЗ
"""
def task2(image, marked_image, watermark, alpha, level):
    rho_scaled_arr = []

    for k in np.arange(0.55, 1.5, 0.15):
        scaled_image = scale(marked_image, k)

        extracted_watermark = extracting(marked_image, image, alpha, level)
        extracted_watermark_scaled = extracting(scaled_image, image, alpha, level)

        rho = get_rho(watermark, extracted_watermark)
        rho_scaled = get_rho(watermark, extracted_watermark_scaled)

        rho_scaled_arr.append(rho_scaled if rho_scaled > 0 else 0)

        if k == 0.55:
            print(f"Rho: {rho}")

        if np.abs(k - 0.85) < 0.001 or np.abs(k - 1.3) < 0.001:
            fig = plt.figure()
            plt.suptitle(f"k={k}")
            sp = fig.add_subplot(1, 2, 1)
            sp.set_title("Исходное изображение")
            imshow(marked_image, cmap='gray', vmin=0, vmax=255)

            sp = fig.add_subplot(1, 2, 2)
            sp.set_title("Масштабированное изображение")
            imshow(scaled_image, cmap='gray', vmin=0, vmax=255)

    plt.figure()
    plt.title("Зависимость меры близости от k")
    plt.plot(np.arange(0.55, 1.5, 0.15), rho_scaled_arr)


"""
Медианная фильтрация
Результат:
ЦВЗ не извлекается, мера близости около 0.01
ДВП (Хаара) не устойчиво к преобразованию "Медианная фильтрация",
поскольку ...,
что делает невозможным корректное извлечение ЦВЗ
"""
def task3(image, marked_image, watermark, alpha, level):
    rho_median_arr = []

    for m in np.arange(3, 16, 2):
        median_image = median_filter(marked_image, (m, m))

        extracted_watermark = extracting(marked_image, image, alpha, level)
        extracted_watermark_median = extracting(median_image, image, alpha, level)

        rho = get_rho(watermark, extracted_watermark)
        rho_median = get_rho(watermark, extracted_watermark_median)

        rho_median_arr.append(rho_median if rho_median > 0 else 0)

        if m == 3:
            print(f"Rho: {rho}")

        if m == 3 or m == 13:
            fig = plt.figure()
            plt.suptitle(f"m={m}")
            sp = fig.add_subplot(1, 2, 1)
            sp.set_title("Исходное изображение")
            imshow(marked_image, cmap='gray', vmin=0, vmax=255)

            sp = fig.add_subplot(1, 2, 2)
            sp.set_title("Изображение после медианной фильтрации")
            imshow(median_image, cmap='gray', vmin=0, vmax=255)

    plt.figure()
    plt.title("Зависимость меры близости от m")
    plt.plot(np.arange(3, 16, 2), rho_median_arr)


"""
JPEG
Результат:
ЦВЗ извлекается, мера близости около 0.4
ДВП (Хаара) устойчиво к преобразованию "JPEG" (чем выше качество, тем выше мера близости),
поскольку положение пикселей изображения не изменяется, а значения яркости отклоняются на величину ошибки квантования,
(таблица квантования задается коэффициентом качества QF),
что делает возможным корректное извлечение ЦВЗ
"""
def task4(image, marked_image, watermark, alpha, level):
    rho_jpeg_arr = []

    for qf in np.arange(30, 95, 10):
        imsave("./images/marked_image.jpg", marked_image, quality=qf)
        jpeg_image = imread("./images/marked_image.jpg")

        extracted_watermark = extracting(marked_image, image, alpha, level)
        extracted_watermark_jpeg = extracting(jpeg_image, image, alpha, level)

        rho = get_rho(watermark, extracted_watermark)
        rho_jpeg = get_rho(watermark, extracted_watermark_jpeg)

        rho_jpeg_arr.append(rho_jpeg if rho_jpeg > 0 else 0)

        if qf == 30:
            print(f"Rho: {rho}")

        if qf == 50 or qf == 80:
            fig = plt.figure()
            plt.suptitle(f"QF={qf}")
            sp = fig.add_subplot(1, 2, 1)
            sp.set_title("Исходное изображение")
            imshow(marked_image, cmap='gray', vmin=0, vmax=255)

            sp = fig.add_subplot(1, 2, 2)
            sp.set_title("Изображение после JPEG сжатия")
            imshow(jpeg_image, cmap='gray', vmin=0, vmax=255)

    plt.figure()
    plt.title("Зависимость меры близости от QF")
    plt.plot(np.arange(30, 95, 10), rho_jpeg_arr)


def main():
    level = 3
    alpha = 0.45
    key = 321
    modified = True

    image = read_image("./images/bridge.tif")

    marked_image, watermark = embedding(image, key, alpha, level, modified)
    print(np.count_nonzero(marked_image - image) / (512 * 512))
    # imsave("./images/marked_image.png", marked_image)
    # marked_image = read_image("./images/marked_image.png")

    task1(image, marked_image, watermark, alpha, level)
    task2(image, marked_image, watermark, alpha, level)
    task3(image, marked_image, watermark, alpha, level)
    task4(image, marked_image, watermark, alpha, level)


    show()




main()