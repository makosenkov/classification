from PIL import Image, ImageDraw
from google.cloud import vision
from google.cloud.vision import types
import os, cv2 as cv
import numpy as np
import re


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"

client = vision.ImageAnnotatorClient()


def draw_bound(img, block):
    bound = to_nd_array(block.bounding_box)
    cv.drawContours(img, [bound], 0, (0, 0, 255), 2)


def get_passport_block_text(img, block_type, dict):
    img_copy = img.copy()
    image = types.Image(content=cv.imencode('.jpg', img)[1].tostring())
    response = client.document_text_detection(image=image, image_context={"language_hints": ["ru"]})
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            # print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                if paragraph.confidence >= 0.6:
                    # print('Paragraph confidence: {}'.format(
                    #     paragraph.confidence))

                    for word in paragraph.words:
                        if word.confidence >= 0.6:
                            word_text = ''.join([
                                symbol.text for symbol in word.symbols
                            ])
                            # print('Word text: {} (confidence: {})'.format(
                            #     word_text, word.confidence))
                            if block_type == 'header':
                                if re.search("[а-я]+", word_text) or word_text.lower() == 'федерация' or word_text.lower() == 'российская':
                                    continue
                                elif re.match("[0-9]{1,2}[-\., ]{1,2}[0-9]{1,2}[-\., ]{1,2}[0-9]{4}", word_text):
                                    text = re.sub(r"[^\d]", ".", word_text)
                                    dict['issue_date'] = text
                                elif re.match("[0-9]{3}-[0-9]{3}", word_text):
                                    dict['code'] = word_text
                                else:
                                    dict['issuer'] += word_text + ' '

                            elif block_type == 'bottom':
                                if re.match("^[\d]+$", word_text) or re.match("^[А-Я]{1}[а-я]+", word_text) or re.match("^[A-Za-z<>]+", word_text):
                                    continue
                                elif re.match("[0-9]{1,2}[-\., ]{1,2}[0-9]{1,2}[-\., ]{1,2}[0-9]{4}", word_text):
                                    text = re.sub(r"[^\d]", ".", word_text)
                                    dict['birthdate'] = text
                                elif re.match("(ЖЕН|МУЖ)[.,]?", word_text.upper()):
                                    dict['sex'] = word_text
                                elif dict['birthdate'] is '' and dict['sex'] is '':
                                    dict['fio'] += word_text + ' '
                                else:
                                    dict['birthplace'] += word_text + ' '

                            else:
                                if re.match("^[\d]+$", word_text):
                                    dict['number'] += word_text

                            draw_bound(img_copy, word)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


    cv.imshow("bounds", img_copy)
    cv.waitKey()
    cv.destroyAllWindows()


def to_nd_array(box):
    bound = np.array([[box.vertices[0].x, box.vertices[0].y],
                      [box.vertices[1].x, box.vertices[1].y],
                      [box.vertices[2].x, box.vertices[2].y],
                      [box.vertices[3].x, box.vertices[3].y]])
    return bound


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
    return passport_data