import cv2

sample = cv2.imread('../resources/img/extractedimages/Waldo_4.png')

#face = sample[300:500, 800:1000]

x, y, w, h = cv2.selectROI("Select", sample)

face = sample[y: y + h, x: x + w]

cv2.imshow("Face", face)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("CroppedTemplate.png", face)