import re
import cv2
import pytesseract
from pytesseract import Output

#path for tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

img_ = cv2.imread("./images/text/image_text_1.jpg", cv2.IMREAD_ANYCOLOR)
gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

print("Converted RGB image to grayscale...")
print("Resizing image to 350x350 scale...")

# Resizing image to 1080x1080 scale
img_ = cv2.resize(gray,(1000, 950))
print("Resized...")

# Write image with name 'saved_img-final.jpg' in folder
img = cv2.imwrite(filename='.\images\saved_img-final.jpg', img=img_)

img = cv2.imread('.\images\saved_img-final.jpg')

d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 10:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)