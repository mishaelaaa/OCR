# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
from matplotlib import pyplot as plt
import numpy as np
import argparse, imutils, time, cv2, pytesseract

#path for tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

#Creating argument dictionary for the default arguments needed in the code. 
args = {"video"         : "cap",
        "image"         : "img_",
        "east"          : "frozen_east_text_detection.pb", 
        "min_confidence": 0.5, 
        "width"         : 320, 
        "height"        : 320}

'''
# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
'''
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

'''
# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
'''

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	#vs = VideoStream(src=0).start()
	vs = VideoStream(src=1).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1)
    vs = cap

# start the FPS throughput estimator
fps = FPS().start()

while True:
    try:
        check, frame = vs.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)

        # Waits for a pressed key.
        key = cv2.waitKey(1)

        # If I pressed a S-key.
        if key == ord('s'): 
            # Write image with name 'saved_img.jpg' in folder
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            vs.release()
            # Read image with name 'saved_img.jpg' in folder
            img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            # view the image 
            img_new = cv2.imshow("Captured Image", img_new)

            # Waits for a pressed S-key.
            cv2.waitKey(1650) 
            cv2.destroyAllWindows()

            print("Processing image...")
            
            # Read image with name 'saved_img.jpg' in folder
            img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            print("Converting RGB image to grayscale...")
            # Converting RGB image to grayscale
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

            print("Converted RGB image to grayscale...")
            print("Resizing image to 350x350 scale...")

            # Resizing image to 350x350 scale
            img_ = cv2.resize(gray,(350, 350))
            print("Resized...")

            # Write image with name 'saved_img-final.jpg' in folder
            img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
            print("Image saved!")
        
            break
        
        # If I pressed a Q-key.
        elif key == ord('q'):
            #close the camera
            print("Turning off camera.")
            vs.release()

            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    # close the all windows 
    except(KeyboardInterrupt):
        print("Turning off camera.")
        vs.release()

        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

args['image']='saved_img.jpg'
image = cv2.imread(args['image'])

#Saving a original image and shape
orig = image.copy()
(origH, origW) = image.shape[:2]

# set the new height and width to default 320 by using args #dictionary.  
(newW, newH) = (args["width"], args["height"])

#Calculate the ratio between original and new image for both height and weight. 
#This ratio will be used to translate bounding box location on the original image. 
rW = origW / float(newW)
rH = origH / float(newH)

# resize the original image to new dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# construct a blob from the image to forward pass it to EAST model
blob = cv2.dnn.blobFromImage(image, 0.5, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)

# load the pre-trained EAST model for text detection 
net = cv2.dnn.readNet(args["east"])

'''
# We would like to get two outputs from the EAST model. 
    1. Probabilty scores for the region whether that contains text or not. 
    2. Geometry of the text -- Coordinates of the bounding box detecting a text
# The following two layer need to pulled from EAST model for achieving this. 
'''
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

#Forward pass the blob from the image to get the desired output layers
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

## Returns a bounding box and probability score if it is more than minimum confidence
def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	# loop over rows
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# loop over the number of columns
		for i in range(0, numC):
			if scoresData[i] < args["min_confidence"]:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# extracting the rotation angle for the prediction and computing the sine and cosine
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# using the geo volume to get the dimensions of the bounding box
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# compute start and end for the text pred box
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	# return bounding boxes and associated confidence_val
	return (boxes, confidence_val)

 # Find predictions and  apply non-maxima suppression
(boxes, confidence_val) = predictions(scores, geometry)
boxes = non_max_suppression(np.array(boxes), probs=confidence_val)

##Text Detection and Recognition 
# initialize the list of results
results = []

# loop over the bounding boxes to find the coordinate of bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	#extract the region of interest
	r = orig[startY:endY, startX:endX]

	#configuration setting to convert image to string.  
	configuration = ("-l eng --psm 8")
    ##This will recognize the text from the image of bounding box
	text = pytesseract.image_to_string(r, config=configuration)

	# append bbox coordinate and associated text to the list of results 
	results.append(((startX, startY, endX, endY), text))

#Display the image with bounding box and recognized text
orig_image = orig.copy()

''' -c CONFIGVAR=VALUE

Set value for parameter CONFIGVAR to VALUE. Multiple -c arguments are allowed.
'''

''' --dpi N

Specify the resolution N in DPI for the input image(s). A typical value for N is 300. Without this option, the resolution is read from the metadata included in the image. If an image does not include that information, Tesseract tries to guess it.
'''

''' -l LANG 

[link] : https://github.com/tesseract-ocr/tessdata
'''

''' -l SCRIPT

The language or script to use. If none is specified, eng (English) is assumed. Multiple languages may be specified, separated by plus characters. Tesseract uses 3-character ISO 639-2 language codes (see LANGUAGES AND SCRIPTS).
'''

''' --psm N

Set Tesseract to only run a subset of layout analysis and assume a certain form of image. The options for N are:

0       Orientation and script detection (OSD) only.
1       Automatic page segmentation with OSD.
2       Automatic page segmentation, but no OSD, or OCR. (not implemented)
3       Fully automatic page segmentation, but no OSD. (Default)
4       Assume a single column of text of variable sizes.
5       Assume a single uniform block of vertically aligned text.
6       Assume a single uniform block of text.
7       Treat the image as a single text line.
8       Treat the image as a single word.
9       Treat the image as a single word in a circle.
10      Treat the image as a single character.
11      Sparse text. Find as much text as possible in no particular order.
12      Sparse text with OSD.
13      Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
'''

''' --oem N

Specify OCR Engine mode. The options for N are:

0       Original Tesseract only.
1       Neural nets LSTM only.
2       Tesseract + LSTM.
3       Default, based on what is available.
'''

''' --tessdata-dir PATH

Specify the location of tessdata path.
'''

''' --user-patterns FILE

Specify the location of user patterns file.
'''

''' --user-words FILE

Specify the location of user words file.
'''

# Moving over the results and display on the image
for ((start_X, start_Y, end_X, end_Y), text) in results:
	# display the text detected by Tesseract
	print("{}\n".format(text))

	# Displaying text
	text = "".join([x if ord(x) > 10 else "" for x in text]).strip()
	cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
		(0, 0, 255), 2)
	cv2.putText(orig_image, text, (start_X, start_Y - 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)

plt.imshow(orig_image)
plt.title('Output')
plt.show()
