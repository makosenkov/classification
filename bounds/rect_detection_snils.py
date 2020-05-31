import numpy as np
import cv2 as cv


def findBiggest(contours):
    biggest = np.array([])
    biggest_area = 0
    for cnt in contours:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        heigth, width = sidesOfBox(box)
        aspect_ratio = heigth / width
        if 0.4 < aspect_ratio < 2.3:
            if 10000 < area < 600000 and area > biggest_area:
                biggest = box
                biggest_area = area
                print(aspect_ratio)

    return biggest


def sidesOfBox(box):
    height = np.sqrt(np.power(box[0][0] - box[3][0], 2) + np.power(box[0][1] - box[3][1], 2))
    width = np.sqrt(np.power(box[0][0] - box[1][0], 2) + np.power(box[0][1] - box[1][1], 2))
    return height, width


def adjustContrast(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return final


def showContours(path):
    fn = path  # имя файла, который будем анализировать
    img = cv.imread(fn)
    img = cv.resize(img, (680, 880))
    if img is None:
        print('Could not open or find the image:', path)
        exit(0)
    img = adjustContrast(img)
    cv.imshow("Contrasted", img)
    row, col = img.shape[:2]
    bottom = img[row - 2:row, 0:col]
    mean = cv.mean(bottom)[0]
    bordersize = 10
    border = cv.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )
    img = border
    cv.imshow('bordered', img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(img, 5)
    # cv.imshow('blur', blur)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 2)
    cv.imshow('thresh', thresh)
    # canny_output = cv.Canny(thresh, 100, 200)
    # cv.imshow('Canny', canny_output)
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    biggest = findBiggest(contours0)
    if biggest.size != 0:
        cv.drawContours(img, [biggest], 0, (0, 0, 255), 2)

    cv.imshow('contours', img)  # вывод обработанного кадра в окно

    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    showContours('imgs/snils415.jpg')
    showContours('imgs/snils472.jpg')
    showContours('imgs/snils553.jpg')
