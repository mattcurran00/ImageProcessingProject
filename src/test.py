
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('waldo.jpg')
template = cv2.imread('template.png')

# Convert both images to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


THRESHOLD = 0.30

# Compare the template against the image
result = cv2.matchTemplate(
    img_gray, template_gray, cv2.TM_CCOEFF_NORMED
)

h, w = template_gray.shape

# Find the best matching locations
ys, xs = np.where(result >= THRESHOLD)

# If no matches reach the threshold, use the best 10 locations
if len(xs) == 0:
    flat = result.flatten()
    best = np.argsort(flat)[-10:]

    ys, xs = np.unravel_index(best, result.shape)

# Sort guesses by similarity
scores = result[ys, xs]
order = np.argsort(scores)[::-1]

matches = []
for i in order:
    x, y = int(xs[i]), int(ys[i])
    score = float(scores[i])

    # Skip guesses too close to an existing box
    if all(abs(x - mx) > w // 2 or abs(y - my) > h // 2
           for mx, my, ms in matches):
        matches.append((x, y, score))

    if len(matches) == 10:
        break

# Draw the guesses
output = img.copy()

for x, y, score in matches:
    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(output, f'{score:.2f}', (x, max(y-5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

print(f'Showing {len(matches)} guesses')
print(f'Best score: {result.max():.3f}')

plt.figure(figsize=(14, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Grayscale Template Matching - Best Guesses')
plt.axis('off')
plt.show()
