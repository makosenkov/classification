import os
import difflib
import bounds.rect_detection_snils as snils


def extract_text(path):
    cropped = snils.showContours(path)
    snils_data = snils.get_text(cropped)
    return snils_data


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
    for i in range(len(images)):
        path = directory + images[i]
        snils_data = extract_text(path)
        extracted_number = snils_data['number']
        true_number = numbers[i]
        if extracted_number != true_number:
            not_equal_counter += 1
            error_image_names.append(images[i])
            # print('{} => {}'.format(extracted_number, true_number))
            diff_count = 0
            for i, s in enumerate(difflib.ndiff(extracted_number, true_number)):
                if s[0] == ' ':
                    continue
                else:
                    diff_count += 1
            diff_counts.append(diff_count)
        print("Progress: " + str(counter) + "/" + str(len(numbers)))
        counter += 1

    some_symbol_errors = 0
    for i in range(len(diff_counts)):
        if diff_counts[i] == 1 or diff_counts[i] == 2:
            some_symbol_errors += 1

    print("Ошибок в 1-2 символах: " + str(some_symbol_errors) + "/" + str(len(numbers)))
    print("Ошибок в большем количестве символов: " + str(not_equal_counter) + "/" + str(len(numbers)))
