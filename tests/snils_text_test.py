import os
import difflib

import numpy as np
import cv2
import bounds.rect_detection_snils as snils


def extract_text(path):
    cropped = snils.showContours(path)
    if cv2.countNonZero(cropped) == 0:
        snils_data = {'number': ''}
    else:
        snils_data = snils.get_text(cropped)
    return snils_data


def distance(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current_row = range(n + 1)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]


if __name__ == '__main__':
    snils_dir_test = '../data/test/snils/images/'
    snils_file_test = '../data/test/snils/numbers.txt'
    file = open(snils_file_test)
    directory = snils_dir_test
    images = os.listdir(directory)

    filenames = []
    imgs = []
    numbers = [line.rstrip('\n') for line in file]
    counter = 0
    not_equal_counter = 0
    error_image_names = []
    diff_counts = []
    diff_count_levenstein = 0
    symbols = 0
    unreadable = 0
    for i in range(len(images)):
        path = directory + images[i]
        snils_data = extract_text(path)
        extracted_number = snils_data['number']
        if extracted_number == '' :
            unreadable += 1
        else:
            true_number = numbers[i]
            diff_count_levenstein += distance(extracted_number, true_number)
            symbols += len(true_number)

        print("Progress: " + str(counter) + "/" + str(len(numbers)))
        counter += 1


    print("Точность по Левенштейну: " + str(1 - diff_count_levenstein / symbols))
    # print("Ошибок в 1-2 символах: " + str(some_symbol_errors) + "/" + str(len(numbers)))
    print("Нечитаемые: " + str(unreadable) + "/" + str(len(numbers)))