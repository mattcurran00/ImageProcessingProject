import cv2
import matplotlib as plt


class ColorSpaceIsolation():

    def getHSVPositions():
        """Will return an array of x:x and y: values where possible waldo is"""
        return []

    def displayWaldosHSV(self, image):
        """Just displays highlighted Characters"""
        mask = cv2.inRange(image, (0, 0, 120), (80, 80, 255))  
        highlighted_red = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("image", highlighted_red)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    colorSpaceIsolation = ColorSpaceIsolation()
    I = cv2.imread("../img/beach.jpeg")

    colorSpaceIsolation.displayWaldosHSV(I)

