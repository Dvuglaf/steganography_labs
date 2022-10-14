"""
�������:
1. ����������� ��������� ��� ?? ��� ��������������� ������������������ �������� ����� �� �����,
������������� �� ����������� ������. ����� 1/2 ��������� �� 3 ������.

2. ����������� ������������� ��������� ���������� � ������������ ��������� -
��� (�������� �����), 3 ������ ������������

3. ����������� ����������� ���������� ����������������� ������� � ��������� HH �������.

4. ������������ �������� ���������� ��� ������ ��������� �������������� �� ������� ��������� � ��������� �������.
��������� ��� �� ����

5. ������� �������� ���������� �� ����� � �������� ��������� �. 2 ��� �������� ����������.

6. ������������ ������ ����������� ��� ??? �������� ������� (�� ���� � �������������� �������
��������� ��������� ����������); ��������� �������������� ��� ������ ������� �������� ??(??, ???)

7. ����������� �������������� ������ �������� ��������� ����������� ������� �������� � ����� �����������
???????? > 30 �� (��� �� ������� ������ ��������, �������� 30 ��), ��� ���� ����������
����� ����������, ��������������� ����������� �������� ??.

8. ��������� �������������� ������������ ���������� ������� ����������� ���������� - �Beta: Laplace�
"""

import numpy as np
import pywt
import skimage.metrics
import scipy
from skimage.io import imread, imshow, show, imsave
from matplotlib import pyplot as plt


def read_image(image_path):
    return np.array(imread(image_path)).astype(np.uint8)


# ������ ������� �������������� ����� � �������� ������� ������������
def dwt(matrix: np.array, level: int):
    return pywt.wavedec2(matrix, 'haar', level=level)


# �������� ������� �������������� �����
def idwt(wavelet_coeffs: np.array):
    return pywt.waverec2(wavelet_coeffs, 'haar')


# ��������� ��������� ��������������� ������� � seed'��
def generate_watermark(size: int, key: int, mean=0., spread=1.):
    rng = np.random.default_rng(key)
    return rng.normal(mean, spread, size)


# ���� ��������
def get_rho(watermark: np.array, extracted_watermark: np.array):
    return np.cumsum(watermark * extracted_watermark)[-1] / \
           np.sqrt(np.cumsum(watermark ** 2)[-1] * np.cumsum(extracted_watermark ** 2)[-1])


def get_beta(matrix: np.array):
    mask = [[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]]  # ���������
    convolved = scipy.signal.convolve2d(matrix, mask, mode='same', boundary='symm', fillvalue=0)
    convolved = np.abs(convolved)  # ���������� �� ������������� ��������

    return convolved / np.max(convolved)


# ��������� ������ ��� ��������� ��������, ��������� �� ������� �����
def get_threshold(matrix: np.array):
    sorted_matrix = np.sort(matrix, axis=None).reshape((matrix.shape[0], matrix.shape[1]))
    return sorted_matrix[(matrix.shape[0] - 1) // 2, (matrix.shape[1] - 1) - 1]


# �������������� ������� � ������ ����������, ���������� ��� ��������������� ������� �������
def get_matrix_diagonals(matrix: np.array):
    diagonals_count = matrix.shape[0] + matrix.shape[1] - 1

    horizontal_flip = np.fliplr(matrix)  # ��������� ����� ��, ������� ������-���� � ��������
    vertical_flip = np.flipud(matrix)  # ��������� ����� ��, ������� �����-����� � ��������

    diagonals = []

    # � ������ ����������� �� diagonals_count//2 ����������, ������� �� ���
    for i in range(-(diagonals_count // 2), diagonals_count // 2 + 1):
        if i % 2 == 0:  # �������� ��������� (�� ������)
            diagonals.append(list(horizontal_flip.diagonal(-i)))
        else:  # ������ ��������� (�� ������)
            diagonals.append(list(vertical_flip.diagonal(i)))

    return diagonals


# �������������� ������� ���������� � �������
def diagonals_to_matrix(diagonals: np.array):
    diagonals_count = len(diagonals)

    matrix = np.zeros((diagonals_count // 2 + 1, diagonals_count // 2 + 1))

    horizontal_flip = np.fliplr(matrix)  # ��������� ����� ��, ������� ������-���� � ������-������ � ��������
    vertical_flip = np.flipud(matrix)  # ��������� ����� ��, ������� �����-����� � �����-������� � ��������

    # � ������ ����������� �� diagonals_count//2 ����������, ������� �� ���
    for i in range(-(diagonals_count // 2), diagonals_count // 2 + 1):
        if i % 2 == 0:  # �������� ��������� (�� ������)
            if i > 0:
                np.fill_diagonal(horizontal_flip[i:, :],
                                 diagonals[i + diagonals_count // 2])  # ����������� ��� ������� (������� � i = 1)
            else:  # ������������� �������� ������������ �������� ��������� (������� � ����� ����� �������)
                np.fill_diagonal(horizontal_flip[:, -i:],
                                 diagonals[i + diagonals_count // 2])  # ����������� ��������
        else:  # ������ ��������� (�� ������)
            if i > 0:
                np.fill_diagonal(vertical_flip[:, i:],
                                 diagonals[i + diagonals_count // 2])  # ����������� ��������
            else:  # ������������� �������� ������������ �������� ��������� (������� � ����� ����� �������)
                np.fill_diagonal(vertical_flip[-i:, ],
                                 diagonals[i + diagonals_count // 2])  # ����������� ��������

    return matrix


def embedding(image: np.array, key: int, alpha: float, level: int, modified: bool):
    wavelet_coeffs = dwt(image, level)
    hh_zone = wavelet_coeffs[1][2]

    watermark_size = hh_zone.shape[0] * hh_zone.shape[1] // 2
    watermark = generate_watermark(watermark_size, key)

    diagonals = get_matrix_diagonals(hh_zone)

    threshold = get_threshold(hh_zone)
    w_i = 0  # ��� ������� �� ������� watermark

    # ������ �� ��������� ���������� ����������, ������� �� 2-�� �� ������ ���������
    for i in range(1, len(diagonals)):  # ����� ���� ����������
        for j in range(len(diagonals[i])):  # ����� ��������� � ���������
            if np.absolute(diagonals[i][j]) > np.absolute(threshold):
                diagonals[i][j] *= (1 + alpha * watermark[w_i])
                w_i += 1
                if w_i == watermark_size:
                    break
        else:
            continue  # ���������� � ����� ������ ��������, ���� �� ���� break �� ���������� �����
        break  # ���������� ������ � ������ break ���������� �����

    marked_hh_zone = diagonals_to_matrix(diagonals)

    # ��-�� �������������� tuple ���������� ������ ���� � list, ������ � ��� �������� � ����� ��������� �������
    wavelet_coeffs_list = list(wavelet_coeffs[1])
    wavelet_coeffs_list[2] = marked_hh_zone
    wavelet_coeffs.pop(1)  # �������� ��������� tuple
    wavelet_coeffs_tuple = tuple(wavelet_coeffs_list)
    wavelet_coeffs.insert(1, wavelet_coeffs_tuple)  # ������� ����������� tuple

    if not modified:  # ������� �����������
        return idwt(wavelet_coeffs), watermark
    else:  # ���������������� ����������� � ���������� beta
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

    # ������ �� ��������� ���������� ����������, ������� �� 2-�� (�� ������) ���������
    for i in range(1, len(diagonals_original)):  # ����� ���� ����������
        for j in range(len(diagonals_original[i])):  # ����� ��������� ���������
            if np.absolute(diagonals_original[i][j]) > np.absolute(threshold):
                watermark[w_i] = (diagonals_marked[i][j] - diagonals_original[i][j]) / \
                                 (alpha * diagonals_original[i][j])
                w_i += 1
                if w_i == watermark_size:
                    return watermark

    return watermark


# ��������� �������� alpha, ��� ������� rho �����������
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
    sp.set_title("�������� �����������")
    imshow(image, cmap='gray')

    sp = fig.add_subplot(1, 2, 2)
    sp.set_title("����������� � ���")
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
    level = 3  # ������� ������������
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
