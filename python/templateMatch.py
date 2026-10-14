import cv2
import numpy as np

stock = cv2.imread('../resources/img/extractedimages/Waldo_6.png')
template = cv2.imread('../resources/img/templates/CroppedTemplate.png')

best_score = -1
best_location = None
best_size = None

# Try 5 different template scales
scales = [0.5, 0.6, 0.7, 0.8, 0.9]

threshold = 0.55
scales = [0.5, 0.6, 0.7, 0.8, 0.9]

for scale in scales:
    resized = cv2.resize(template, None, fx=scale, fy=scale)
    h, w = resized.shape[:2]

    if h > stock.shape[0] or w > stock.shape[1]:
        continue

    result = cv2.matchTemplate(
        stock, resized, cv2.TM_CCOEFF_NORMED
    )

    locations = np.where(result >= threshold)

    print(f"Scale {scale}: {len(locations[0])} matches")

    for pt in zip(*locations[::-1]):
        cv2.rectangle(stock, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2 )

# Draw rectangle around best match
if best_location is not None:
    x, y = best_location
    w, h = best_size

    cv2.rectangle(stock, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(f"Best confidence: {best_score:.3f}")

# Display resized result
display = cv2.resize(stock, None, fx=0.5, fy=0.5)
cv2.imshow("Waldo Detection", display)
cv2.waitKey(0)
cv2.destroyAllWindows()