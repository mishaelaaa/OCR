import cv2
import numpy as np
import tensorflow as tf

args = {"image":"./images/technical/"}

args['image']="./images/technical/45.jpg"
image = cv2.imread(args['image'], cv2.IMREAD_UNCHANGED)

scale_percent = 35 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape) 

cv2.imshow("image", resized)
print(image.shape)

height, width, _ = resized.shape

#cutting image 
roi = resized[30: height, 20:width]

roi = tf.random.normal(shape=(256, 256, 3))
tf.image.adjust_contrast(roi,2)

cv2.imshow("roi", roi)
print(roi.shape)

cv2.waitKey(0)