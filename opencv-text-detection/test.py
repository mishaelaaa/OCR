'''
import requests

url = 'https://app.nanonets.com/api/v2/OCR/Model/c10ef23a-2bd4-4686-80ec-f65a22b20cca/LabelFile/'

data = {'file': open('19.jpg', 'rb')}

response = requests.post(url, auth=requests.auth.HTTPBasicAuth('klcPSAi_kEnZDeL1TMeWIWWzv1TubFzl', ''), files=data)

print(response.text)
'''

'''
import cv2, os, argparse, pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'

# construct the argument parser and parse the arguments
args = {"image":"./images/", 
        "preprocess" : "thresh" }

#args['image']="./images/1.5.png"
#args['image']="./images/Untitled.png"
#args['image']="./images/technical_1.jpg"
args['image']="./images/technical_1_cut.jpg"
#args['image']="./images/technical_2.jpg" 

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
'''
	
# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# construct the argument parser and parse the arguments
args = {"image":"./images/technical_1_cut.jpg", 
        "east":"frozen_east_text_detection.pb", 
        "min_confidence":0.5, 
        "width":320, 
        "height":320,
        "padding" : 0.0,
        "preprocess" : "default"}

# load the input image and grab the image dimensions
args['image']="./images/technical_1_cut.jpg"
image = cv2.imread(args["image"])
#image = cv2.resize(image,(360,480))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

orig = image.copy()
(origH, origW) = image.shape[:2]
 
# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)
 
# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested in -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
 
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
 
# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# initialize the list of results
results = []
 
# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# in order to obtain a better OCR of the text we can potentially
	# apply a bit of padding surrounding the bounding box -- here we
	# are computing the deltas in both the x and y directions
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	# apply padding to each side of the bounding box, respectively
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

    # extract the actual padded ROI
	roi = orig[startY:endY, startX:endX]
	
	# in order to apply Tesseract v4 to OCR text we must supply
	# (1) a language, (2) an OEM flag of 4, indicating that the we
	# wish to use the LSTM neural net model for OCR, and finally
	# (3) an OEM value, in this case, 7 which implies that we are
	# treating the ROI as a single line of text
	config_eng = ("-l eng --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config_eng)
	config_bul = ("-l bul --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config_bul)
 
	# add the bounding box coordinates and OCR'd text to the list
	# of results
	results.append(((startX, startY, endX, endY), text))

# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])
 
# loop over the results
for ((startX, startY, endX, endY), text) in results:
	# display the text OCR'd by Tesseract
	print("OCR TEXT")
	print("========")
	print("{}\n".format(text))
 
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV, then draw the text and a bounding box surrounding
	# the text region of the input image
	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv2.putText(output, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
 
	# show the output image
	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)