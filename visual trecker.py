import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def preprocess(image, window=True):
    """
    Предварительная обработка изображения

    1 этап: использование функции log для увеличения чувствительности к тёмным
    оттенкам
    2 этап: нормализация до среднего арифметического 0 и нормы 1
    3 этап: добавление виньетки
    """
    image = image.astype('float32')

    # Логарифм для увеличения чувствительности к тёмным оттенкам
    image = np.log(1 + image)

    # Нормализация до среднего арифметического 0 и нормы 1
    mu = np.mean(image)
    sigma = np.std(image - mu)
    image = (image - mu) / sigma

    if window:
        def sin2dwin(z):
            """ Создаёт маску 'виньетки' для изображения """
            xaxis = np.arange(z.shape[0])
            yaxis = np.arange(z.shape[1])
            x, y = np.meshgrid(xaxis, yaxis)
            wx = np.pi / x.shape[0]
            wy = np.pi / x.shape[1]
            return np.sin(wx * x) * np.sin(wy * y)

        # Добавляем 'виньетку'
        window = sin2dwin(image)
        image *= window

    return image



def get_desired_correl(image_resolution, sigma=2.0):
    """
    Генерирует g с плотным (sigma=2.0) пиком гаусовой формы в центре
    корреляционной матрицы
    """
    xaxis = np.arange(image_resolution[0])
    yaxis = np.arange(image_resolution[1])
    x, y = np.meshgrid(xaxis, yaxis)

    center = 0.5 * image_resolution
    norm1 = 1 / np.sqrt(2 * np.pi * sigma)
    norm2 = 1 / (2 * sigma ** 2)
    desired_correl = norm1 * np.exp(-(np.square(x - center[0]) + \
        np.square(y - center[1])) * norm2)

    return desired_correl


def get_filter_conj(frame):
    """ Вычисляет комплексное сопряжение фильтра """
    resolution = np.array(frame.shape)
    # получаем желаемую корреляцию
    desired_correl = get_desired_correl(resolution)
    desired_correl_fft = np.fft.fft2(desired_correl)
    frame_fft = np.fft.fft2(frame)
    # нашли комплексное сопряжение фильтра
    filter_conj_fft = desired_correl_fft / frame_fft

    return filter_conj_fft


def main():
    # создаём объект захвата видео
    cap = cv2.VideoCapture('test video 464x464.mp4')
    # создаём объект записи видео
    correlation_writer = cv2.VideoWriter(
    filename='correlation_video.mp4',
    fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
    fps=30,
    frameSize=(464, 464)
    )
    # создаём объект записи видео
    capture_writer = cv2.VideoWriter(
    filename='capture_video.mp4',
    fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
    fps=30,
    frameSize=(464, 464)
    )

    # получаем комплексное сопряжение фильтра для первого кадра
    retval, frame = cap.read()  # читаем первый кадр
    # конвертируем его в оттенки серого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    SHAPE = gray_frame.shape
    CENTER = [SHAPE[0] // 2, SHAPE[1] // 2]  # [232, 232]
    position = CENTER.copy()
    # вычисляем комплексное сопряжение фильтра
    filter_conj_fft = get_filter_conj(gray_frame)

    # создаём объект захвата видео
    cap = cv2.VideoCapture('test video 464x464.mp4')
    shift = [0, 0]
    shifts = []
    while True:
        retval, frame = cap.read()
        if retval:
            # читаем кадр
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # вычисляем FFT кадра
            frame_fft = np.fft.fft2(gray_frame)
            # находим корреляционную матрицу в FFT области
            correl_fft = frame_fft * filter_conj_fft
            # находим корреляционную матрицу с помощью IFFT
            correl = np.fft.ifft2(correl_fft)
            # находим абсолютные значения для отрисовки матрицы
            correl_view = 255 * np.abs(correl)
            correl_view = correl_view.astype('uint8')
            
            # преобразуем в RGB
            correl_view_bgr = cv2.cvtColor(correl_view, cv2.COLOR_GRAY2BGR)
            # записываем картинку в видео
            correlation_writer.write(correl_view_bgr)
            
            peak = np.unravel_index(correl_view.argmax(), SHAPE)
            shift[0] = peak[0] - CENTER[0]
            shift[1] = peak[1] - CENTER[1]
            shifts.append(shift)
            
            position[0] += shift[1]
            position[1] += shift[0]
            cv2.circle(frame, center=position, radius=10, color=(0, 0, 255))
            capture_writer.write(frame)

            filter_conj_fft = get_filter_conj(gray_frame)
        else:
            correlation_writer.release()
            capture_writer.release()
            break


if __name__ == "__main__":
    main()
