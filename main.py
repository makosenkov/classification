import argparse
import pprint

from tensorflow.python.keras.models import load_model

import convnets.utilities.doc_prediction as dp
import bounds.rect_detection_passport as passport
import bounds.rect_detection_snils as snils
import ocr.text_detection_google as ocr_google
import ocr.text_detection_tesseract as tesseract


def extract_text(img, type):
    if type == 'passport':
        cropped = passport.showContours(img)
        passport_data = passport.get_text(cropped)
        pprint.pprint(passport_data)
    elif type == 'snils':
        cropped = snils.showContours(img)
        snils_data = snils.get_text(cropped)
        pprint.pprint(snils_data)
    else:
        print("Данный тип документа не поддерживает распознавание реквизитов")


parser = argparse.ArgumentParser(description='Скрипт для всех частей системы.')
parser.add_argument('--doctype', help='Тип документа, варианты: passport, snils, dover, ndfl, sved',
                    required=True)
parser.add_argument('--image', help='Путь к изображению', required=True)
args = parser.parse_args()
type = args.doctype
weights_path = 'data/weights/' + type + '_model_mcp.h5'

img = args.image
prediction = dp.predict_one(weights_path, img, type)
if not prediction:
    print('Вероятно, вы приложили документ неверного типа. Проверьте, пожалуйста, приложение')
    exit(0)
else:
    print('Тип документа, скорее всего, корректен, идем дальше')
extract_text(img, type)




