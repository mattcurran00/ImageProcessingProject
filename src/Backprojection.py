import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the scene and a small crop of Waldo (the model)
scene = cv2.imread("../resources/img/extractedimages/Waldo_4.png")
model = cv2.imread("../resources/img/templates/fullbody.png")

h, w, d = scene.shape
scene = cv2.resize(scene, dsize=(w//2, h//2))

# Convert both to HSV
scene_hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
model_hsv = cv2.cvtColor(model, cv2.COLOR_BGR2HSV)

# Model histogram on Hue and Saturation (2D), quantized into 30 and 32 bins
M = cv2.calcHist([model_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])

# Normalize so values sit in 0-255
cv2.normalize(M, M, 0, 255, cv2.NORM_MINMAX)

# Backproject the model histogram onto the scene
back = cv2.calcBackProject([scene_hsv], [0, 1], M, [0, 180, 0, 256], scale=1)

# Smooth the result so locations show up as peaks
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
back = cv2.filter2D(back, -1, disc)

# Threshold to keep the strongest responses
_, mask = cv2.threshold(back, 50, 255, cv2.THRESH_BINARY)

# Find the single strongest peak
_, _, _, max_loc = cv2.minMaxLoc(back)
cv2.circle(scene, max_loc, 30, (0, 255, 0), 3)

cv2.imshow("backprojection", back)
cv2.imshow("result", scene)
cv2.waitKey(0)