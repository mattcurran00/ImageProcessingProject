import cv2
import matplotlib as plt
import numpy as np
import os

class ColorSpaceIsolation():

    def getHSVPositions():
        """Will return an array of x:x and y:y values where possible waldo is"""
        return []

    def _red_mask(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask0 = cv2.inRange(image_hsv, np.array([0, 50, 50]),   np.array([10, 255, 255]))
        mask1 = cv2.inRange(image_hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        return cv2.bitwise_or(mask0, mask1)

    def displayWaldosHSV(self, image):
        """Displays the image in grayscale with red regions kept in color."""
        mask = self._red_mask(image)

        # Grayscale, converted back to 3 channels so it can be combined with color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Color only where mask is on; gray only where mask is off
        red_part  = cv2.bitwise_and(image, image, mask=mask)
        gray_part = cv2.bitwise_and(gray_bgr, gray_bgr, mask=cv2.bitwise_not(mask))
        result = cv2.bitwise_or(red_part, gray_part)

        h, w = result.shape[:2]
        result = cv2.resize(result, (w // 2, h // 2))

        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    colorSpaceIsolation = ColorSpaceIsolation()
    I = cv2.imread("../img/medieval.jpg")

    colorSpaceIsolation.displayWaldosHSV(I)

