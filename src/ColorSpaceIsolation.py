import cv2
import matplotlib as plt
import numpy as np
import os

class ColorSpaceIsolation():

    def getHSVPositions(self, image):
        """Return (x, y, w, h) boxes where Waldo might be."""
        _, possible_locations = self._filter_mask(image, self._waldos_mask(image))
        return possible_locations
    
    def _waldos_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red = cv2.inRange(hsv, (170, 100, 80), (179, 255, 255))
        white = cv2.inRange(hsv, (0, 0, 190), (179, 65, 255))
        blue = cv2.inRange(hsv, (95, 100, 70), (115, 255, 255))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Red pixels that are within ~2px of white = stripes
        stripes = cv2.bitwise_and(red, cv2.dilate(white, kernel))

        # Only keep white that is close to a stripe (not every cloud or page)
        white_near_red = cv2.bitwise_and(white, cv2.dilate(stripes, kernel))

        return stripes | white_near_red | blue

    def _filter_mask(self, image, mask):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red = cv2.inRange(
            hsv, np.array([170, 100, 100]), np.array([179, 255, 255])
        )

        blue = cv2.inRange(
            hsv, np.array([95, 100, 70]), np.array([115, 255, 255])
        )

        # Close small gaps between colour fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = np.zeros_like(mask)

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            if area < 30 or h < 10 or w < 3:
                continue

            # Count red and blue pixels inside this candidate region
            red_count = cv2.countNonZero(red[y:y+h, x:x+w])
            blue_count = cv2.countNonZero(blue[y:y+h, x:x+w])

            # Require both shirt red and trouser blue
            if red_count < 5 or blue_count < 5:
                continue

            # Reject very wide regions
            if h / float(w) < 0.8:
                continue

            cv2.drawContours(
                result, [contour], -1, 255, cv2.FILLED
            )

            boxes.append((x, y, w, h))

        return result, boxes



    def displayWaldosHSV(self, image):
        """Displays the image in grayscale with red regions kept in color."""
        mask = self._waldos_mask(image)
        mask, possible_locations = self._filter_mask(image, mask)

        # Grayscale, converted back to 3 channels so it can be combined with color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Color only where mask is on; gray only where mask is off
        red_part  = cv2.bitwise_and(image, image, mask=mask)
        gray_part = cv2.bitwise_and(gray_bgr, gray_bgr, mask=cv2.bitwise_not(mask))
        result = cv2.bitwise_or(red_part, gray_part)

        for (x,y,w,h) in possible_locations:
            cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),2)

        h, w = result.shape[:2]
        original = cv2.resize(image, (w // 2, h // 2))
        result = cv2.resize(result, (w // 2, h // 2))
        mask = cv2.resize(mask, (w // 2, h //2))

        cv2.imshow("original", original)
        cv2.imshow("mask", mask)
        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


if __name__ == "__main__":
    colorSpaceIsolation = ColorSpaceIsolation()
    I = cv2.imread("../resources/img/extractedimages/Waldo_6.png")

    colorSpaceIsolation.displayWaldosHSV(I)

