import pytesseract
from pytesseract import Output
import cv2
import re
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def get_passport_block_text(image, block_type, dict):
    results = pytesseract.image_to_data(image, lang='rus', config='--psm 12', output_type=Output.DICT)
    result_str = ""
    for i in range(0, len(results["text"])):

        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]

        word_text = results["text"][i]
        conf = int(results["conf"][i])
        if conf > 40:
            if block_type == 'header':
                if re.search("[а-я%]+",
                             word_text) or re.search("фед|ция|фел|ссий|ская", word_text.lower()):
                    continue
                elif re.match("[0-9]{1,2}[-\., ]{1,2}[0-9]{1,2}[-\., ]{1,2}[0-9]{4}", word_text):
                    text = re.sub(r"[^\d]", ".", word_text)
                    dict['issue_date'] = text
                elif re.match("[0-9]{3}-[0-9]{3}", word_text):
                    dict['code'] = word_text
                elif re.search("[^а-яА-Я0-9.\-№]", word_text):
                    continue
                else:
                    dict['issuer'] += word_text + ' '

            elif block_type == 'bottom':
                if re.match("^[\d]+$", word_text) or re.search("[а-я]+", word_text) or re.match("^[A-Za-z<>]+", word_text):
                    continue
                elif re.match("[0-9]{1,2}[-\., ]{1,2}[0-9]{1,2}[-\., ]{1,2}[0-9]{4}", word_text):
                    text = re.sub(r"[^\d]", ".", word_text)
                    dict['birthdate'] = text
                elif re.search("[^а-яА-Я0-9\.]", word_text):
                    continue
                elif re.match("(ЖЕН|МУЖ)[.,]?", word_text.upper()):
                    dict['sex'] = word_text
                elif dict['birthdate'] is '' and dict['sex'] is '':
                    dict['fio'] += word_text + ' '
                else:
                    dict['birthplace'] += word_text + ' '

            else:
                if re.match("^[\d]+$", word_text):
                    dict['number'] += word_text


            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(result_str)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def get_passport_texts(header, bottom, number):
    passport_data = {'issuer': '',
                     'issue_date': '',
                     'code': '',
                     'fio': '',
                     'sex': '',
                     'birthdate': '',
                     'birthplace': '',
                     'number': ''}
    get_passport_block_text(header, 'header', passport_data)
    get_passport_block_text(bottom, 'bottom', passport_data)
    get_passport_block_text(number, 'number', passport_data)
    passport_data['issuer'] = passport_data['issuer'].strip()
    passport_data['fio'] = passport_data['fio'].strip()
    passport_data['birthplace'] = passport_data['birthplace'].strip()
    return passport_data


def get_snils_texts(image):
    results = pytesseract.image_to_data(image, lang='rus', config='--psm 12', output_type=Output.DICT)
    result_str = ""
    for i in range(0, len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        text = results["text"][i]
        conf = int(results["conf"][i])
        if conf > 40:
            if re.match("[\d-]+", text):
                result_str += text
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    result_str = re.sub(r"[^\d]", "", result_str)
    if len(result_str) > 11:
        result_str = result_str[:11]
    dict = {'number': result_str}
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dict