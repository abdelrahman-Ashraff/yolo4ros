# USAGE
# python3 yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import matplotlib.pyplot as plt
import math
from numpy import loadtxt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])

# Resized to match BEV i/p img specifications
image = cv2.resize(image, (680, 420))
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.2f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []



## BEV related props
img_size = (680, 420) # Input image size
pixel_per_meter_y= 97.85714285714286
pixel_per_meter_x= 243.5897435897436
dead_zone = 0.7
M = loadtxt('BEV.csv', delimiter=',')
img = image.copy()

# Getting the warped center
## The bottom's center pt is the location of the camera sensor
## In pixels
original_center = np.array([[[img_size[0]/2,img_size[1]]]],dtype=np.float32)
warped_center = cv2.perspectiveTransform(original_center, M)[0][0] 
warped_center[1] = warped_center[1] + dead_zone * pixel_per_meter_y

## In Meters
warped_center_x = warped_center[0] / pixel_per_meter_x
warped_center_y = warped_center[1] / pixel_per_meter_y + dead_zone

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# Output list holding [eucledian distance, angle w.r.t camera] of each object
output_list = []

# Getting detected object angle
def getAngle(a, b = (warped_center), c = (warped_center[0], warped_center[1] - 1)):
	ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
	return ang

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# extract the middle bottom coordinates
		obj_x = x + int(w/2)
		obj_y = y + int(h)

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

		# draw a circle on the middle bottom
		cv2.circle(img, (obj_x, obj_y), 4, color = (0, 0, 0), thickness = 2)

		# type a text on the image
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

		## Transforming into BEV coordinates
		object_loc = np.array([[[obj_x,obj_y]]],dtype=np.float32)
		object_loc_warped = cv2.perspectiveTransform(object_loc, M)[0][0]

		## Getting object angle w.r.t camera
		ang = getAngle(object_loc_warped)

		if object_loc_warped[0] < warped_center[0]:
			ang = - ang
		else:
			ang = abs(ang)

		## Converting to radians
		ang_rad = math.radians(ang)

		# Getting object distance w.r.t camera
		## Meters representation of detected object
		object_loc_warped[0] = object_loc_warped[0] / pixel_per_meter_x
		object_loc_warped[1] = object_loc_warped[1] / pixel_per_meter_y + dead_zone
		
		## Getting the (dy, dx) between pts coordinates (in meters)
		location_y = object_loc_warped[1] - warped_center_y
		location_x = object_loc_warped[0] - warped_center_x

		## Calculating the Eucledian distance
		distance = math.sqrt((location_x)**2 + (location_y)**2)

		# Testing on real image
		# ang = getAngle((obj_x, obj_y), b = (340, 420), c= (340, 400))
		# if obj_x < img_size[0]/2:
		# 	ang = -ang
		# else:
		# 	ang = abs(ang)

		# Adding the Final results
		output_list.append([distance, ang_rad])

print('Your detected Objects are:\n', output_list)

# Converting into 1D arr for ROS msg purposes
## Preventing concatenating empty list
if output_list:
	output_array = np.concatenate([np.array(i) for i in output_list])
else:
	output_array = np.array(output_list)


############### Robot Operating System ##############
# -- Under Development
# 
# 
# 
# 
# 
# 
# 
# 
####################################################

### Visualization Part -- Under Development

## Getting warped_img
# warped_img = cv2.warpPerspective(img, M,img_size) # Image warping
# warped_img = cv2.resize(warped_img,(640, 420)) # Output image size

# fig = plt.figure()
# plt.figure(figsize=(10,10))
# ax = plt.gca()

# plt1 = fig.add_subplot(121)
# plt2 = fig.add_subplot(122)

## >> Figure 1
# # image = cv2.resize(image, (680, 420))
# # img = cv2.resize(img,(680, 420))
# c1 = plt1.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
# 				vmin = 0, vmax = 5)
# plt1.set_title('Original Image', fontweight='heavy')
# plt1.axis('off')

## >> Figure 2
# print(warped_img.shape)
# c2 = plt2.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB),
# 			   vmin = 0, vmax = 5, cmap ='Blues')
# plt2.set_title('BEV Image', fontweight='heavy')
# plt2.axis('off')

# divider = make_axes_locatable(plt2)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(c2, cax=cax)

## View Figures
# fig.colorbar(c1, orientation='vertical')
# plt.tight_layout()
# plt.show()

## Save the output
# name = args["image"].split("/")[1][:-4] + '_output.jpg'
# cv2.imwrite('output/' + name, concatenated_image)