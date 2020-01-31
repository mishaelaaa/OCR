#import requests

#url = 'https://app.nanonets.com/api/v2/OCR/Model/c10ef23a-2bd4-4686-80ec-f65a22b20cca/LabelFile/'

#data = {'file': open('19.jpg', 'rb')}

#response = requests.post(url, auth=requests.auth.HTTPBasicAuth('klcPSAi_kEnZDeL1TMeWIWWzv1TubFzl', ''), files=data)

#print(response.text)
 
############################################################################################################################

import cv2, os, argparse, pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'

# construct the argument parser and parse the arguments
args = {"image":"./images/", 
        "preprocess" : "thresh" }

#args['image']="./images/1.jpg"
#args['image']="./images/2.jpg"
#args['image']="./images/3.jpg"
#args['image']="./images/4.jpg"
#args['image']="./images/5.jpg"
#args['image']="./images/6.jpg"
#args['image']="./images/7.jpg"
#args['image']="./images/9.jpg"

#args['image']="./images/15.jpg"

#args['image']="./images/14.jpg"
#args['image']="./images/13.jpg"

#args['image']="./images/1.1.png"
#args['image']="./images/1.2.png"
#args['image']="./images/1.3.png"
#args['image']="./images/1.4.png"
#args['image']="./images/1.5.png"
args['image']="./images/Untitled.png"

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
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

#text = pytesseract.image_to_string(filename, lang = 'bul')
text = pytesseract.image_to_string(filename, lang = 'eng')
#text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
 
# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)

############################################################################################################################


#from PIL import Image
#import pytesseract, cv2

#pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'

#img = cv2.imread('13.png',0)
#im = Image.open("13.png")

#text = pytesseract.image_to_string(img, lang = 'eng')

#print(text)

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()