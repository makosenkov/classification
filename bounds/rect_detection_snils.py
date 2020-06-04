import os
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
    imgBordered = utils.border_image(crop_img, [255, 255, 255])
    imgContrasted = utils.adjustContrast(imgBordered)
    imgGray = cv.cvtColor(imgContrasted, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(imgGray, (5, 5), 1)
    canny_output = cv.Canny(blur, 100, 200)
    firstDilated = utils.dilate(canny_output, 6)
    contours0, hierarchy = cv.findContours(firstDilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    imgWithoutSmallContours = utils.removeSmallContours(contours0, firstDilated)
    secondDilated = utils.dilate(imgWithoutSmallContours, 3)
    contours1, hierarchy = cv.findContours(secondDilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    biggest = utils.findBestRectangle(contours1, 0.4, 2.3, 30000, 300000)
    if biggest.size != 0:

        cv.drawContours(img, [biggest], 0, (0, 0, 255), 2)

    cv.imshow('contours', img)  # вывод обработанного кадра в окно

    cv.waitKey()
    cv.destroyAllWindows()
    steps = [imgContrasted, imgGray, blur, canny_output, firstDilated, imgWithoutSmallContours, secondDilated, img]
    labels = ["Contrasted", "Greyscale", "Blurred", "Canny edges", "First dilation", "Removed small contours", "Second dilation", "Bound rectangle"]
    # utils.plot_all_steps(steps, labels)


if __name__ == '__main__':
    snils_dir = '/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/'
    directory = snils_dir
    images = os.listdir(directory)

    filenames = []
    imgs = []
    for name in images:
        path = directory + name
        showContours(path)

    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils511.jpg')
