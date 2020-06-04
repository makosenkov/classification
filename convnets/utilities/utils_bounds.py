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


def border_image(img, color):
    bordersize = 15
    # color = get_dominant_color(img)
    # color = tuple([int(x) for x in color])
    border = cv.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv.BORDER_CONSTANT,
        value=color
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
    cols = 5
    axes = []
    fig = plt.figure(figsize=(19.2, 10.8))

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


def get_dominant_color(a):
    a2D = a.reshape(-1, a.shape[-1])
    col_range = (256, 256, 256)  # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # rotate the warped image if horizontal
    heigth, width, channel = warped.shape
    if width > heigth:
        warped = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)
    return warped


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def thresholding(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 21)
    return thresh
