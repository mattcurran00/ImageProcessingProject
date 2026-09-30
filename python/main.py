import cv2
import numpy as np

def main():
    print('Hello')
    
    template = cv2.imread('../resources/img/templates/template.png')
    target = cv2.imread('../resources/img/samples/beach.jpeg')

    #copy and graysclae
    originalTarget = target.copy()
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    originalTemplate = template.copy()
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    width, height = template.shape[::-1]

    result =  cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5

    location = np.where(result >= threshold)

    for pt in zip(*location[::-1]):
        cv2.rectangle(originalTarget, (pt, pt[0] + width, pt[1] + height), (0, 255, 255), 2)

    cv2.imshow("ResultTemplate",originalTarget)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()