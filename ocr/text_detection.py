import pytesseract
from pytesseract import Output
import cv2


def get_gext(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_data(rgb, lang='rus', config='--psm 12', output_type=Output.DICT)
    result_str = ""
    for i in range(0, len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        # extract the OCR text itself along with the confidence of the
        # text localization
        text = results["text"][i]
        conf = int(results["conf"][i])
        if conf > 40:
            # display the confidence and text to our terminal
            print("Confidence: {}".format(conf))
            print("Text: {}".format(text))
            print("")
            result_str += " " + text
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show the output image
    print(result_str)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()