import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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
                # print(aspect_ratio)
    # print("ratio:" + str(biggest_ratio))
    return biggest


def border_image(img):
    bordersize = 15
    border = cv.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv.BORDER_CONSTANT,
        value=[254, 254, 254]
    )
    img = border
    return img


def dilate(img, iter):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    imgDial = cv.dilate(img, kernel, iterations=iter)  # APPLY DILATION
    return imgDial


def removeSmallContours(contours, image):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    for cnt in contours:
        if cv.arcLength(cnt, True) < 300:
            cv.drawContours(mask, [cnt], -1, 0, -1)
    image = cv.bitwise_and(image, image, mask=mask)
    return image

def plot_all_steps(imgs, labels):
    rows = 2
    cols = 4
    axes = []
    fig = plt.figure(figsize=(19.2,10.8))

    for i in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, i + 1))
        subplot_title = (labels[i])
        axes[-1].set_title(subplot_title)
        plt.axis('off')
        if i in (0, 7):
            plt.imshow(imgs[i])
        else:
            plt.imshow(imgs[i], cmap='Greys_r')
    fig.tight_layout()
    plt.show()