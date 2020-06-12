import os
import cv2 as cv
from convnets.utilities import utils_bounds as utils
import numpy as np
import ocr.text_detection as ocr
from bounds.noteshrink import noteshrink

img_heigth = 1000
img_width = 1400


def add_contours(img):
    start_point_issuer = (15, 50)
    end_point_issuer = (600, 235)
    color = (255, 0, 0)
    thickness = 2
    cropped_rect = img[start_point_issuer[1]:end_point_issuer[1], start_point_issuer[0]:end_point_issuer[0]]
    # image = cv.rectangle(img.copy(), start_point_issuer, end_point_issuer, color, thickness)
    return cropped_rect


def showContours(path):
    fn = path  # имя файла, который будем анализировать
    img = cv.imread(fn)
    if img is None:
        print('Could not open or find the image:', path)
        exit(0)
    blankImage = np.zeros((img_heigth, img_width, 3), np.uint8)
    img = cv.resize(img, (img_heigth, img_width))

    # imgContrasted = utils.adjustContrast(img)

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    imgContrasted = utils.unsharp_image(imgGray)
    # blur = cv.GaussianBlur(imgContrasted, (5, 5), 1)
    canny_output = cv.Canny(imgContrasted, 100, 200)
    crop_img = canny_output[30:img_width - 30, 30:img_heigth - 30].copy()

    withoutBorders = utils.border_image(crop_img, [0, 0, 0], 30)
    firstDilated = utils.dilate(withoutBorders, 4)

    contours0, hierarchy = cv.findContours(firstDilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    imgWithoutSmallContours = utils.removeSmallContours(contours0, firstDilated)
    # eroded = utils.erode(imgWithoutSmallContours, 1)
    secondDilated = utils.dilate(imgWithoutSmallContours, 2)
    contours1, hierarchy = cv.findContours(secondDilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img, contours1, -1, (0, 0, 255), 2)
    biggest = utils.findBestRectangle(contours1, 0.5, 1.9, img_width, img_heigth)
    warpedThresholded = blankImage
    warped = blankImage
    issuer_area = blankImage
    if biggest.size != 0:
        cv.drawContours(img, [biggest], 0, (0, 0, 255), 2)
        warped = utils.four_point_transform(img, biggest)
        resized = cv.resize(warped, (625, 875))
        issuer_area = add_contours(resized)
        # issuer_area_resized = cv.resize(issuer_area, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        # issuer_area_contr = utils.adjustContrast(issuer_area_resized)
        # blurred = cv.bilateralFilter(issuer_area_resized, 2, 127, 127)
        unsharpened = utils.unsharp_text_area(issuer_area)
        options = dict(quiet=False,
                       value_threshold=25,
                       sat_threshold=20,
                       num_colors=8,
                       sample_fraction=5,
                       white_bg=True,
                       saturate=True)
        print(options["sample_fraction"])
        print(options)
        warpedThresholded = noteshrink.notescan(unsharpened, options)
        # warpedGray = cv.cvtColor(issuer_area, cv.COLOR_BGR2GRAY)
        # warpedThresholded = noteshrink.illumination_thresholding(issuer_area)
        # normalized = utils.normalized(unsharpened)
        # warpedThresholded = cv.adaptiveThreshold(warpedGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 17)
        # opened = utils.closing(warpedThresholded)

    # cv.imshow('contours', img)  # вывод обработанного кадра в окно
    cv.imshow("warped", warpedThresholded)
    # cv.imshow("warpedThresh", warpedThresholded)
    cv.waitKey()
    cv.destroyAllWindows()

    # steps = [imgGray,  imgContrasted, blur, canny_output, firstDilated, imgWithoutSmallContours, secondDilated, img,
    #          blankImage, blankImage]
    # labels = ["Contrasted", "Greyscale", "Blurred", "Canny edges", "First dilation", "Removed small contours",
    #           "Second dilation", "Bound rectangle", "Warped", "Thresholded"]
    # utils.plot_all_steps(steps, labels)
    return warpedThresholded


if __name__ == '__main__':
    dir = '/home/mksnkv/Documents/classification/passport_2class_divided_clean/evaluation/passport/'
    directory = dir
    images = os.listdir(directory)

    filenames = []
    imgs = []
    for name in images:
        path = directory + name
        transformed = showContours(path)
        ocr.get_gext(transformed)

    # transformed = showContours('imgs/passport6.jpg')
    # ocr.get_gext(transformed)
