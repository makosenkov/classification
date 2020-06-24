import os
import cv2 as cv
import bounds.utils_bounds as utils


def showContours(path):
    fn = path  # имя файла, который будем анализировать
    img = cv.imread(fn)
    if img is None:
        print('Could not open or find the image:', path)
        exit(0)
    img = cv.resize(img, (700, 900))

    imgContrasted = utils.unsharp_image(img)
    imgGray = cv.cvtColor(imgContrasted, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(imgGray, (5, 5), 1)
    canny_output = cv.Canny(blur, 100, 200)
    firstDilated = utils.dilate(canny_output, 3)
    crop_img = firstDilated[30:870, 30:670].copy()
    withoutBorders = utils.border_image(crop_img, [0,0,0])
    contours0, hierarchy = cv.findContours(withoutBorders.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    imgWithoutSmallContours = utils.removeSmallContours(contours0, withoutBorders)
    secondDilated = utils.dilate(imgWithoutSmallContours, 2)
    contours1, hierarchy = cv.findContours(secondDilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    biggest = utils.findBestRectangle(contours1, 0.4, 2.1, 300000, 600000)
    if biggest.size != 0:

        cv.drawContours(img, [biggest], 0, (0, 0, 255), 2)

    cv.imshow('contours', img)  # вывод обработанного кадра в окно

    cv.waitKey()
    cv.destroyAllWindows()
    steps = [imgContrasted, imgGray, blur, canny_output, firstDilated, imgWithoutSmallContours, secondDilated, img]
    labels = ["Contrasted", "Greyscale", "Blurred", "Canny edges", "First dilation", "Removed small contours", "Second dilation", "Bound rectangle"]
    utils.plot_all_steps(steps, labels)


if __name__ == '__main__':
    dir = '/home/mksnkv/Documents/classification/sved_2class_divided_clean/evaluation/sved/'
    directory = dir
    images = os.listdir(directory)

    filenames = []
    imgs = []
    for name in images:
        path = directory + name
        showContours(path)

    # showContours('/home/mksnkv/Documents/classification/passport_2class_divided_clean/evaluation/passport/passport1548.jpg')
