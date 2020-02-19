import numpy as np 
import glob
import cv2

files = glob.glob('segmented_data/*.jpg')

shape = []

for f in files:
	img = cv2.imread(f)
	h, w, c = img.shape
	shape.append([w, h])
shape = np.array(shape)

w = shape[..., 0]
h = shape[..., 1]

mean_w = np.mean(w)
mean_h = np.mean(h)

print(mean_w, mean_h)
# (96.42311770943796, 29.922587486744433)
# so we choose shape of image is 100 x 30