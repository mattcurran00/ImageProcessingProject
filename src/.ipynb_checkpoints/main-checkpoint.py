import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

# Read and convert main image to grayscale
img_rgb = cv2.imread('image.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read and convert template to grayscale
template = cv2.imread('template.png', 0)  

plt.imshow(template, cmap='gray')
plt.show()