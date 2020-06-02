import numpy as np
import cv2 as cv

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


def sidesOfBox(box):
    height = np.sqrt(np.power(box[0][0] - box[3][0], 2) + np.power(box[0][1] - box[3][1], 2))
    width = np.sqrt(np.power(box[0][0] - box[1][0], 2) + np.power(box[0][1] - box[1][1], 2))
    return height, width


def findBestRectangle(contours, ratio_lb, ratio_rb, area_lb, area_rb):
    biggest = np.array([])
    biggest_area = 0
    biggest_ratio = 0
    for cnt in contours:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        heigth, width = sidesOfBox(box)
        area = heigth * width  # вычисление площади
        aspect_ratio = heigth / width
        if ratio_lb < aspect_ratio < ratio_rb:
            if area_lb < area < area_rb and area > biggest_area:
                biggest = box
                biggest_ratio = aspect_ratio
                biggest_area = area
                print(aspect_ratio)
    print("ratio:" + str(biggest_ratio))
    return biggest


def border_image(img):
    row, col = img.shape[:2]
    bottom = img[row - 2:row, 0:col]
    mean = cv.mean(bottom)[0]
    bordersize = 15
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
    return img


def dilate_erode(img):
    kernel = np.ones((4, 4))

    imgDial = cv.dilate(img, kernel, iterations=6)  # APPLY DILATION
    cv.imshow("dilate", imgDial)
    opening = cv.morphologyEx(imgDial, cv.MORPH_OPEN, kernel)
    # imgThreshold = cv.erode(imgDial, kernel, iterations=2)  # APPLY EROSION
    cv.imshow("erode", opening)
    return opening