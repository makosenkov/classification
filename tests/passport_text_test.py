import json
import os
import cv2
import bounds.rect_detection_passport as passport


def extract_text(path):
    cropped = passport.showContours(path)
    if cv2.countNonZero(cropped) == 0:
        passport_data = {'issuer': '',
                         'issue_date': '',
                         'code': '',
                         'fio': '',
                         'sex': '',
                         'birthdate': '',
                         'birthplace': '',
                         'number': ''}
    else:
        passport_data = passport.get_text(cropped)
    return passport_data


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
    passport_dir_test = '../data/test/passport/images/'
    passport_file_test = '../data/test/passport/text.txt'
    file = open(passport_file_test, encoding="utf8")
    directory = passport_dir_test
    images = os.listdir(directory)

    filenames = []
    imgs = []
    numbers = json.load(file)
    counter = 0
    not_equal_counter = 0

    diff_count_levenstein = 0
    symbols = 0
    unreadable = 0
    for i in range(len(images)):
        path = directory + images[i]
        passport_data = extract_text(path)
        empty_fields = 0
        for key in passport_data:
            if passport_data[key] == '':
                empty_fields += 0
        if empty_fields > 2:
            unreadable += 1
        else:
            true_data = numbers[i]
            diff_count = 0
            for key in passport_data:
                extracted_string = passport_data[key]
                true_string = true_data[key]
                symbols += len(true_string)
                diff_count_levenstein += distance(extracted_string, true_string)
        print("Progress: " + str(counter) + "/" + str(len(numbers)))
        counter += 1

    print("Точность по Левенштейну: " + str(1 - diff_count_levenstein / symbols))
    print("Нечитаемые: " + str(unreadable) + "/" + str(len(numbers)))