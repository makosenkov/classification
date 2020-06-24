from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
import matplotlib.pyplot as plt

rng.seed(12345)


def plot_all_steps(imgs, labels):
    rows = 1
    cols = 5
    axes = []
    fig = plt.figure(figsize=(19.2, 10.8))

    for i in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, i + 1))
        subplot_title = (labels[i])
        axes[-1].set_title(subplot_title)
        plt.axis('off')
        if i == (0, 4):
            plt.imshow(imgs[i])
        else:
            plt.imshow(imgs[i], cmap='Greys_r')
    fig.tight_layout()
    plt.show()


def process():

    # th_adaptive = cv.adaptiveThreshold(src_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 17, 2)
    canny_output = cv.Canny(blur, 100, 200)

    # cv.imshow('Thresholding Adaptive', src_gray)
    # cv.imshow('Canny', canny_output)
    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    contours = sorted(contours, key=cv.contourArea, reverse=True)

    # contour approximation
    for i in contours:
        elip = cv.arcLength(i, True)
        approx = cv.approxPolyDP(i, 0.08 * elip, True)

        if len(approx) == 4:
            doc = approx
            break
    src_copy = src
    # draw contours
    cv.drawContours(src_copy, [doc], -1, (0, 255, 0), 2)
    steps = [src, src_gray, blur, canny_output, src_copy, ]
    labels = ["Image", "Greyscale", "Blurred", "Canny edges", "Contour"]
    plot_all_steps(steps, labels)
    cv.imshow('Contours', src)
    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # for i in range(len(contours)):
    #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #     cv.drawContours(drawing, contours_poly, i, color)
    #     cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
    #                  (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
    #
    # cv.imshow('Contours', drawing)


parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
args = parser.parse_args()
src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
src = cv.resize(src, (700, 900))
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
blur = cv.blur(src_gray, (3, 3))
# cv.imshow('Source', src)
process()
cv.waitKey()
