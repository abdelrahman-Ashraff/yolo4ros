import cv2
import numpy as np
import matplotlib.pyplot as plt

# IMAGE_H = 223
# IMAGE_W = 1280
margin_h = 283
margin_w = 225
width = 640
height = 420
img_size = (width, height)

# src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
# dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
src = np.float32([[258,275], [421,276],[84,420], [594,420]]) #last  for 680*420
dst = np.float32([[margin_w,margin_h],[width-margin_w,margin_h],[margin_w,height],[width-margin_w,height]])

M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

img = cv2.imread('images/frame0052.jpg') # Read the test img

img= cv2.resize(img,(680, 420))

# img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
warped_img = cv2.warpPerspective(img, M,img_size) # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
plt.show()
                                            ############### PROJECT RELATED ###############
