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

"""
Циклический сдвиг
Результат:
ЦВЗ не извлекается, мера близости около 0.02
ДВП (Хаара) не устойчиво к геометрическому преобразованию "Циклический сдвиг",
поскольку оно основано на контурах изображения,
которые, в свою очередь, смещаются при выполнении преобразования,
что делает невозможным корректное извлечение ЦВЗ
"""
def cyclic_shift(image, r):
    N_1, N_2 = image.shape
    shifted = np.roll(image, int(r * N_1), axis=1)
    shifted = np.roll(shifted, int(r * N_2), axis=0)
    return shifted


"""
Масштабирование
Результат:
ЦВЗ не извлекается, мера близости около 0.006
ДВП (Хаара) не устойчиво к геометрическому преобразованию "Масштабирование",
поскольку оно основано на контурах изображения,
которые, в свою очередь, смещаются при выполнении преобразования,
что делает невозможным корректное извлечение ЦВЗ
"""
def scale(image, k):
    N_1, N_2 = image.shape
    scaled_image = rescale(image, k, anti_aliasing=(True if k < 1. else False)) * 255
    N_1_scaled, N_2_scaled = scaled_image.shape
    if k < 1.:
        result_image = np.zeros(image.shape)
        result_image[N_1 // 2 - N_1_scaled // 2: N_1 // 2 + N_1_scaled // 2,
                     N_2 // 2 - N_2_scaled // 2: N_2 // 2 + N_2_scaled // 2] =\
            scaled_image[: N_1_scaled // 2 + N_1_scaled // 2,
                         : N_2_scaled // 2 + N_2_scaled // 2]
        return result_image
    elif k > 1.:
        return scaled_image[N_1_scaled // 2 - N_1 // 2: N_1_scaled // 2 + N_1 // 2,
                            N_2_scaled // 2 - N_2 // 2: N_2_scaled // 2 + N_2 // 2]
    else:
        return image


"""
Медианная фильтрация
Результат:
ЦВЗ не извлекается, мера близости около 0.01
ДВП (Хаара) не устойчиво к преобразованию "Медианная фильтрация",
что делает невозможным корректное извлечение ЦВЗ
"""
def median_filtered(image, m):
    median_image = median_filter(image, (m, m))
    return median_image


"""
JPEG
Результат:
ЦВЗ извлекается, мера близости около 0.4
ДВП (Хаара) устойчиво к преобразованию "JPEG" (чем выше качество, тем выше мера близости),
поскольку положение пикселей изображения не изменяется, а значения яркости отклоняются на величину ошибки квантования,
(таблица квантования задается коэффициентом качества QF),
что делает возможным корректное извлечение ЦВЗ
"""
def jpeg(image, qf):
    imsave("./images/marked_image.jpg", image, quality=qf)
    return imread("./images/marked_image.jpg")


"""
Внесение искажений в изображение со встроенным ЦВЗ, извлечение из модифицированного изображения ЦВЗ
и сравнение его со встраиваемым ЦВЗ по мере близости rho.
param_values задает массив значений параметра искажения, для которого проводится исследование.
Возвращает массив зачений мер близости для каждого значения параметра искажения.
"""
def process_image(image, marked_image, watermark, alpha, level, process_function, param_values):
    rho_values = []

    for value in param_values:
        processed_image = process_function(marked_image, value)

        extracted_watermark = extracting(processed_image, image, alpha, level)
        rho = get_rho(watermark, extracted_watermark)
        rho_values.append(rho if rho > 0 else 0)

    return rho_values


def plot_images(image, processed_image, title):
    fig = plt.figure()

    sp = fig.add_subplot(1, 2, 1)
    sp.set_title("Исходное изображение")
    imshow(image, cmap='gray', vmin=0, vmax=255)

    sp = fig.add_subplot(1, 2, 2)
    sp.set_title(title)
    imshow(processed_image, cmap='gray', vmin=0, vmax=255)


"""
Отображение нескольких графиков.
"""
def plot_graph(x_values, y_values, colors, labels, title, x_label, y_label):
    plt.figure()
    plt.suptitle(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for (x, y, color, label) in zip(x_values, y_values, colors, labels):
        plt.plot(x, y, color=color, label=label)

        if len(x) >= 2:
            step = x[1] - x[0]
            plt.xticks(np.arange(np.min(x), np.max(x) + step, step))

        plt.yticks(np.arange(0, 1.1, 0.1))

    plt.legend()


if __name__ == "__main__":
    level = 3
    alpha = 0.45
    key = 321

    image = read_image("./images/bridge.tif")

    marked_image, watermark = embedding(image, key, alpha, level, False)
    imsave("./images/marked_image.png", marked_image)
    marked_image = read_image("./images/marked_image.png")
    extracted_watermark = extracting(marked_image, image, alpha, level)
    rho = get_rho(watermark, extracted_watermark)

    marked_image_laplace, watermark = embedding(image, key, alpha, level, True)
    imsave("./images/marked_image.png", marked_image_laplace)
    marked_image_laplace = read_image("./images/marked_image.png")
    extracted_watermark_laplace = extracting(marked_image_laplace, image, alpha, level)
    rho_laplace = get_rho(watermark, extracted_watermark_laplace)

    print(f"Rho: {rho}")
    print(f"Rho (Beta: Laplace): {rho_laplace}")
    # print(np.count_nonzero(marked_image - image) / (512 * 512))

    process_functions = [cyclic_shift, scale,
                         median_filtered, jpeg]
    param_values = [np.arange(0., 1, 0.1), np.arange(0.55, 1.5, 0.15),
                    np.arange(3, 16, 2), np.arange(30, 95, 10)]

    colors = ['blue', 'orange']
    labels = ['Обычное встраивание', 'Взвешенное встраивание']
    titles = ['Зависимость меры близости от циклического сдвига на долю r',
              'Зависимость меры близости от коэффициента масштабирования k',
              'Зависимость меры близости от размера окна m',
              'Зависимость меры близости от параметра качества QF']
    x_labels = ['Доля сдвига r', 'Коэффициент масштабирования k',
                'Размера окна m', 'Параметр качества QF']
    y_label = 'Мера близости rho'

    for (param_value, process_function, title, x_label) in zip(param_values, process_functions, titles, x_labels):
        rho_values = process_image(image, marked_image, watermark,
                                   alpha, level, process_function, param_value)
        rho_values_laplace = process_image(image, marked_image_laplace, watermark,
                                           alpha, level, process_function, param_value)

        x_values = [param_value, param_value]
        y_values = [rho_values, rho_values_laplace]

        plot_graph(x_values, y_values, colors, labels, title, x_label, y_label)

    processed_image_example = cyclic_shift(marked_image, 0.6)
    title = 'Циклический сдвиг изображения на долю 0.6'
    plot_images(marked_image, processed_image_example, title)

    processed_image_example = scale(marked_image, 0.7)
    title = 'Масштабирование изображения с коэффициентом 0.7'
    plot_images(marked_image, processed_image_example, title)
    processed_image_example = scale(marked_image, 1.3)
    title = 'Масштабирование изображения с коэффициентом 1.3'
    plot_images(marked_image, processed_image_example, title)

    processed_image_example = median_filtered(marked_image, 11)
    title = 'Медианная фильтрация изображения окном размера (11,11)'
    plot_images(marked_image, processed_image_example, title)

    processed_image_example = jpeg(marked_image, 50)
    title = 'Сжатие изображения в формате JPEG с показателем качества 50'
    plot_images(marked_image, processed_image_example, title)

    show()

