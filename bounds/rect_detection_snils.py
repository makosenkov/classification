import numpy as np
import cv2 as cv

hsv_min = np.array((0, 54, 5), np.uint8)
hsv_max = np.array((187, 255, 253), np.uint8)


def findBiggest(contours):
    biggest = np.array([])
    biggest_area = 0
    for cnt in contours:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        sides = (box[0][0] - box[1][0]) / (box[0][1] - box[3][1])
        # if sides > 0.5 and sides < 3:
        if area > 200 and area < 600000 and area > biggest_area:
            biggest = box
            biggest_area = area


    return biggest

if __name__ == '__main__':
    fn = 'imgs/snils415.jpg'  # имя файла, который будем анализировать
    img = cv.imread(fn)
    img = cv.resize(img, (700, 900))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(img, 5)
    cv.imshow('blur', blur)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 2)
    cv.imshow('thresh', thresh)
    canny_output = cv.Canny(thresh, 100, 200)
    cv.imshow('Canny', canny_output)
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    biggest = findBiggest(contours0)
    if biggest.size != 0:
        cv.drawContours(img, [biggest], 0, (0, 0, 255), 2)

    cv.imshow('contours', img)  # вывод обработанного кадра в окно

    cv.waitKey()
    cv.destroyAllWindows()
