import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('carnival.jpg')
template = cv2.imread('template.png')

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

THRESHOLD = 0.45               
SCALES = np.linspace(0.5, 1.5, 21)  


def find_matches(img_gray, template_gray, threshold, scales):
    boxes = []  # (x, y, w, h, score)
    for scale in scales:
        t = cv2.resize(template_gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
        th, tw = t.shape
        if th > img_gray.shape[0] or tw > img_gray.shape[1]:
            continue

        res = cv2.matchTemplate(img_gray, t, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= threshold)
        for x, y in zip(xs, ys):
            boxes.append((x, y, tw, th, float(res[y, x])))
    return boxes


def non_max_suppression(boxes, overlap=0.3):
    """Merge overlapping detections, keeping the highest-scoring one."""
    if not boxes:
        return []
    rects = [[x, y, w, h] for x, y, w, h, _ in boxes]
    scores = [s for *_, s in boxes]
    idxs = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.0,
                            nms_threshold=overlap)
    return [boxes[i] for i in np.array(idxs).flatten()]


boxes = find_matches(img_gray, template_gray, THRESHOLD, SCALES)
matches = non_max_suppression(boxes)

output = img_rgb.copy()
for x, y, w, h, score in matches:
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(output, f'{score:.2f}', (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

print(f'Found {len(matches)} match(es)')
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()