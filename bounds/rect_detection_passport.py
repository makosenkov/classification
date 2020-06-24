import cv2 as cv
import bounds.utils_bounds as utils
import numpy as np
import ocr.text_detection_tesseract as tesseract
import pprint

img_heigth = 1000
img_width = 1400


def crop_header_block(img):
    start_point = (15, 50)
    end_point = (550, 235)
    return crop_block(img, start_point, end_point)


def crop_bottom_block(img):
    start_point = (200, 405)
    end_point = (550, 750)
    return crop_block(img, start_point, end_point)

def crop_number_block(img):
    start_point = (525, 475)
    end_point = (625, 800)
    return crop_block(img, start_point, end_point)


def crop_block(img, start_point, end_point):
    cropped_rect = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]
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

    imgContrasted = utils.unsharp_image_old(imgGray)
    # blur = cv.GaussianBlur(imgContrasted, (5, 5), 1)
    canny_output = cv.Canny(imgContrasted, 100, 200)
    dilated_with_borders = utils.dilate(canny_output, 4)
    crop_img = canny_output[30:img_width - 30, 30:img_heigth - 30].copy()

    withoutBorders = utils.border_image(crop_img, [0, 0, 0], 30)
    firstDilated = utils.dilate(withoutBorders, 4)

    contours0, hierarchy = cv.findContours(firstDilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    imgWithoutSmallContours = utils.removeSmallContours(contours0, firstDilated)
    # eroded = utils.erode(imgWithoutSmallContours, 1)
    secondDilated = utils.dilate(imgWithoutSmallContours, 2)
    contours1, hierarchy = cv.findContours(secondDilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img, contours1, -1, (0, 0, 255), 2)
    biggest = utils.findBestRectangle(contours1, 0.5, 1.9, img_width, img_heigth, 'passport')
    number_area_unsharpened = blankImage
    bottom_area_unsharpened = blankImage
    issuer_area_unsharpened = blankImage
    resized = blankImage
    if biggest.size != 0:
        cv.drawContours(img, [biggest], 0, (0, 0, 255), 2)
        warped = utils.four_point_transform(imgGray, biggest, 'passport')
        resized = cv.resize(warped, (625, 875))
        issuer_area_unsharpened, bottom_area_unsharpened, number_area_unsharpened = crop_from_resized(resized.copy())
        # issuer_area_contr = utils.adjustContrast(issuer_area_resized)
        # blurred = cv.bilateralFilter(issuer_area_resized, 2, 127, 127)
        # warpedGray = cv.cvtColor(unsharpened, cv.COLOR_BGR2GRAY)
        # warpedThresholded = noteshrink.illumination_thresholding(issuer_area)
        # normalized = utils.normalized(unsharpened)
        # warpedThresholded = cv.adaptiveThreshold(warpedGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 27)
        # opened = utils.closing(warpedThresholded)

    # cv.imshow('contours', img)  # вывод обработанного кадра в окно
    # cv.imshow("warped", resized)
    # cv.imshow("header", issuer_area_unsharpened)
    # cv.imshow("bottom", bottom_area_unsharpened)
    # cv.imshow("warpedThresh", dilated_with_borders)
    # cv.imshow("Greyscale", imgWithoutSmallContours)
    # cv.imshow("Contrasted", imgContrasted)
    # cv.waitKey()
    # cv.destroyAllWindows()
    #
    # steps = [img, imgGray,  imgContrasted, canny_output, firstDilated, imgWithoutSmallContours, img,
    #          resized, issuer_area_unsharpened, bottom_area_unsharpened, number_area_unsharpened, blankImage]
    # labels = ["Image", "Greyscale", "Contrasted", "Canny edges", "First dilation", "Removed small contours",
    #           "Bound rectangle", "Warped", "Header", "Bottom", "Number", "Blank"]
    # utils.plot_all_steps(steps, labels)
    return resized


def crop_from_resized(resized):
    issuer_area = crop_header_block(resized)
    bottom_area = crop_bottom_block(resized)
    number_area = crop_number_block(resized)
    issuer_area_resized = cv.resize(issuer_area, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC)
    bottom_area_resized = cv.resize(bottom_area, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC)
    number_area_resized = cv.resize(number_area, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC)
    issuer_area_unsharpened = utils.unsharp_text_area(issuer_area_resized)
    bottom_area_unsharpened = utils.unsharp_text_area(bottom_area_resized)
    number_area_unsharpened = utils.unsharp_text_area(number_area_resized)
    number_area_unsharpened = cv.rotate(number_area_unsharpened, cv.ROTATE_90_COUNTERCLOCKWISE)
    return issuer_area_unsharpened, bottom_area_unsharpened, number_area_unsharpened


def get_text(img):
    header, bottom, number = crop_from_resized(img.copy())
    passport_data = tesseract.get_passport_texts(header, bottom, number)
    if passport_data['issuer'] is '' or passport_data['birthplace'] is '':
        resized_rotated = cv.rotate(img, cv.ROTATE_180)
        header, bottom, number = crop_from_resized(resized_rotated)
        passport_data = tesseract.get_passport_texts(header, bottom, number)
    return passport_data


if __name__ == '__main__':
    # dir = '../data/test/contours/'
    # directory = dir
    # images = os.listdir(directory)
    #
    # filenames = []
    # imgs = []
    # for name in images:
    #     path = directory + name
    #     resized = showContours(path)
    #     passport_data = get_text(resized)
    #     pprint.pprint(passport_data)

    resized = showContours('../data/test/contours/vivek.jpg')
    passport_data = get_text(resized)
    pprint.pprint(passport_data)
    # transformed = showContours('imgs/passport6.jpg')
    # ocr.get_gext(transformed)
