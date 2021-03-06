import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def sidesOfBox(box):
    height = np.sqrt(np.power(box[0][0] - box[3][0], 2) + np.power(box[0][1] - box[3][1], 2))
    width = np.sqrt(np.power(box[0][0] - box[1][0], 2) + np.power(box[0][1] - box[1][1], 2))
    return height, width


def findBestRectangle(contours, ratio_lb, ratio_rb, img_heigth, img_width, doctype):
    biggest = np.array([])
    biggest_area = 0
    biggest_ratio = 0
    if doctype == 'passport':
        area_lb = 0.15 * img_heigth * img_width
        area_rb = 0.85 * img_heigth * img_width
    elif doctype == 'snils':
        area_lb = 0.1 * img_heigth * img_width
        area_rb = 0.85 * img_heigth * img_width
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


def border_image(img, color, width):
    bordersize = width
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


def erode(img, iter):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    imageEroded = cv.erode(img, kernel, iterations=iter)
    return imageEroded


def removeSmallContours(contours, image):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    for cnt in contours:
        if cv.arcLength(cnt, True) < 300:
            cv.drawContours(mask, [cnt], -1, 0, -1)
    image = cv.bitwise_and(image, image, mask=mask)
    return image


def plot_all_steps(imgs, labels):
    rows = 3
    cols = 4
    axes = []
    fig = plt.figure(figsize=(19.2, 10.8))

    for i in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, i + 1))
        subplot_title = (labels[i])
        axes[-1].set_title(subplot_title)
        plt.axis('off')
        if i == (0, 7, 8, 9, 10, 11):
            plt.imshow(imgs[i])
        else:
            plt.imshow(imgs[i], cmap='Greys_r')
    fig.tight_layout()
    plt.show()


def four_point_transform(image, pts, doctype):
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
    heigth, width = warped.shape
    if doctype == 'passport':
        if width > heigth:
            warped = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)
    elif doctype == 'snils':
        if heigth > width:
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


def unsharp_image(img):
    alpha = 1.3  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)

    unsharped_image = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    # gaussian_3 = cv.GaussianBlur(img, (9, 9), 10.0)
    # unsharped_image = cv.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
    return unsharped_image


def unsharp_image_old(img):

    gaussian_3 = cv.GaussianBlur(img, (9, 9), 10.0)
    unsharped_image = cv.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
    return unsharped_image


def unsharp_text_area(img):
    gaussian_3 = cv.GaussianBlur(img, (3, 3), 10.0)
    unsharp_image = cv.addWeighted(img, 1.6, gaussian_3, -0.4, 0, img)
    return unsharp_image


def closing(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    imageEroded = cv.erode(img, kernel, iterations=2)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    closed = cv.dilate(imageEroded, kernel, iterations=1)  # APPLY DILATION
    return closed


def normalized(img):
    image = cv.normalize(img, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC3)
    return image
