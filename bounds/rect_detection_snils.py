import os
import cv2 as cv
import bounds.utils_bounds as utils
import numpy as np
import ocr.text_detection_tesseract as tesseract

img_heigth = 680
img_width = 880
blankImage = np.zeros((img_heigth, img_width, 1), np.uint8)


def crop_number_block(img):
    start_point = (75, 75)
    end_point = (525, 225)
    cropped_rect = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    return cropped_rect


def showContours(path):
    fn = path  # имя файла, который будем анализировать
    img = cv.imread(fn)
    if img is None:
        print('Could not open or find the image:', path)
        exit(0)
    img = cv.resize(img, (img_heigth, img_width))
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgContrasted = utils.unsharp_image_old(imgGray)
    canny_output = cv.Canny(imgContrasted, 100, 200)
    crop_img = canny_output[30:img_width - 30, 30:img_heigth - 30].copy()
    imgBordered = utils.border_image(crop_img, [0, 0, 0], 30)
    firstDilated = utils.dilate(imgBordered, 4)
    contours0, hierarchy = cv.findContours(firstDilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    imgWithoutSmallContours = utils.removeSmallContours(contours0, firstDilated)
    secondDilated = utils.dilate(imgWithoutSmallContours, 2)
    contours1, hierarchy = cv.findContours(secondDilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    biggest = utils.findBestRectangle(contours1, 0.4, 2.3, img_heigth, img_width, 'snils')
    resized = blankImage
    if biggest.size != 0:

        cv.drawContours(img, [biggest], 0, (0, 0, 255), 2)
        warped = utils.four_point_transform(imgGray, biggest, 'snils')
        resized = cv.resize(warped, (550, 375))



    # steps = [img, imgGray, imgContrasted, canny_output, firstDilated, imgWithoutSmallContours, secondDilated, img,
    #          resized, number_area_unsharpened, blankImage, blankImage, ]
    # labels = ["Image", "Greyscale", "Contrasted", "Canny edges", "First dilation", "Removed small contours",
    #           "Second dilation", "Bound rectangle", "Warped", "Number", "Blank", "Blank"]
    # utils.plot_all_steps(steps, labels)
    return resized


def crop_from_resized(resized):
    number_area = crop_number_block(resized.copy())
    number_area_resized = cv.resize(number_area, None, fx=1.1, fy=1.1, interpolation=cv.INTER_CUBIC)
    number_area_unsharpened = utils.unsharp_text_area(number_area_resized)
    # number_area_unsharpened = utils.unsharp_image(number_area_resized)
    # warpedGray = cv.cvtColor(number_area_unsharpened, cv.COLOR_BGR2GRAY)
    # warpedThresholded = cv.adaptiveThreshold(warpedGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 7)
    # cv.imshow("sdf", number_area_unsharpened)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return number_area_unsharpened


def get_text(img):
    cropped = crop_from_resized(img)
    snils_data = tesseract.get_snils_texts(cropped)

    if len(snils_data['number']) < 9:
        resized_rotated = cv.rotate(img, cv.ROTATE_180)
        cropped = crop_from_resized(resized_rotated)
        snils_data = tesseract.get_snils_texts(cropped)
    return snils_data


if __name__ == '__main__':
    snils_dir = '../data/evaluation/snils/snils/'
    directory = snils_dir
    images = os.listdir(directory)

    filenames = []
    imgs = []
    for name in images:
        path = directory + name
        resized = showContours(path)
        snils_data = get_text(resized)
        print(name + ": " + snils_data)
    # showContours('/home/mksnkv/Documents/classification/snils_2class_divided_clean/evaluation/snils/snils511.jpg')
