import json
import os
import cv2
import bounds.rect_detection_passport as passport


def extract_text(path):
    cropped = passport.showContours(path)
    # if cv2.countNonZero(cropped) == 0:
    #     passport_data = {'issuer': '',
    #                      'issue_date': '',
    #                      'code': '',
    #                      'fio': '',
    #                      'sex': '',
    #                      'birthdate': '',
    #                      'birthplace': '',
    #                      'number': ''}
    # else:
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
    passport_file_test = '../data/test/passport/test_text.txt'
    file = open(passport_file_test, encoding="utf8")
    directory = passport_dir_test
    images = os.listdir(directory)

    filenames = []
    imgs = []
    numbers = json.load(file)
    counter = 0
    not_equal_counter = 0

    diff_count_birthplace = 0
    diff_count_birthdate = 0
    diff_count_issuer = 0
    diff_count_issue_date = 0
    diff_count_code = 0
    diff_count_fio = 0
    diff_count_sex = 0
    diff_count_number = 0
    symbols_birthplace = 0
    symbols_birthdate = 0
    symbols_issuer = 0
    symbols_issue_date = 0
    symbols_code = 0
    symbols_fio = 0
    symbols_sex = 0
    symbols_number = 0
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
            for key in passport_data:
                extracted_string = passport_data[key]
                true_string = true_data[key]
                if key == 'birthplace':
                    symbols_birthplace += len(true_string)
                    diff_count_birthplace += distance(extracted_string, true_string)
                elif key == 'birthdate':
                    symbols_birthdate += len(true_string)
                    diff_count_birthdate += distance(extracted_string, true_string)
                elif key == 'issuer':
                    symbols_issuer += len(true_string)
                    diff_count_issuer += distance(extracted_string, true_string)
                elif key == 'issue_date':
                    symbols_issue_date += len(true_string)
                    diff_count_issue_date += distance(extracted_string, true_string)
                elif key == 'code':
                    symbols_code += len(true_string)
                    diff_count_code += distance(extracted_string, true_string)
                elif key == 'fio':
                    symbols_fio += len(true_string)
                    diff_count_fio += distance(extracted_string, true_string)
                elif key == 'sex':
                    symbols_sex += len(true_string)
                    diff_count_sex += distance(extracted_string, true_string)
                elif key == 'number':
                    symbols_number += len(true_string)
                    diff_count_number += distance(extracted_string, true_string)
        print("Progress: " + str(counter) + "/" + str(len(numbers)))
        counter += 1

    print("Точность даты рождения: " + str(1 - diff_count_birthdate / symbols_birthdate))
    print("Точность места рождения: " + str(1 - diff_count_birthplace / symbols_birthplace))
    print("Точность кем выдан: " + str(1 - diff_count_issuer / symbols_issuer))
    print("Точность даты выдачи: " + str(1 - diff_count_issue_date / symbols_issue_date))
    print("Точность код подразделения: " + str(1 - diff_count_code / symbols_code))
    print("Точность ФИО: " + str(1 - diff_count_fio / symbols_fio))
    print("Точность пол: " + str(1 - diff_count_sex / symbols_sex))
    print("Точность серия/номер: " + str(1 - diff_count_number / symbols_number))
    print("Нечитаемые: " + str(unreadable) + "/" + str(len(numbers)))