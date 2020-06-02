import numpy as np
import cv2 as cv
from convnets.utilities import utils_bounds as utils


def showContours(path):
    fn = path  # имя файла, который будем анализировать
    img = cv.imread(fn)
    if img is None:
        print('Could not open or find the image:', path)
        exit(0)
    img = cv.resize(img, (680, 880))
    crop_img = img[30:850, 30:650].copy()
    cv.imshow("cropped", crop_img)
    imgContrasted = utils.adjustContrast(crop_img)
    # cv.imshow("Contrasted", img)
    imgBordered = utils.border_image(imgContrasted)
    # cv.imshow('bordered', img)
    imgGray = cv.cvtColor(imgBordered, cv.COLOR_BGR2GRAY)
    # blur = cv.medianBlur(img, 5)
    blur = cv.GaussianBlur(imgGray, (5, 5), 1)
    # cv.imshow('blur', blur)
    # thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 2)
    # cv.imshow('thresh', thresh)
    canny_output = cv.Canny(blur, 100, 200)
    cv.imshow('Canny', canny_output)
    thresh = utils.dilate_erode(canny_output)

    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img, contours0, -1, (0, 0, 255), 2)
    biggest = utils.findBestRectangle(contours0, 0.4, 2.3, 10000, 600000)
    if biggest.size != 0:
        cv.drawContours(img, [biggest], 0, (0, 0, 255), 2)

    cv.imshow('contours', img)  # вывод обработанного кадра в окно

    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils381.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils382.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils386.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils387.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils388.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils389.jpg')
    showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils393.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils394.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils395.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils396.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils400.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils401.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils402.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils403.jpg')
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils404.jpg')
