import cv2, os, argparse, pytesseract
import numpy as np
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'
# construct the argument parser and parse the arguments
args = {"image":"./images/technical/", 
        "east":"frozen_east_text_detection.pb", 
        "min_confidence":1, 
        "width":150, 
        "height":150,
        "preprocess" : "thresh"}

args['image']="./images/technical/1.jpg"
image = cv2.imread(args["image"])

# load the example image and convert it to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 10)
 
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "name_new_image.png".format(os.getpid())
cv2.imwrite(filename, gray)
# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary 
text = pytesseract.image_to_string(filename, lang = 'bul')
text = pytesseract.image_to_string(filename, lang = 'eng')
#text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
 
# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
 
cv2.destroyAllWindows()