import numpy as np
import cv2
import glob
from utils import cut_img_bbox, show_img, save_img
import os

OUTPUT_DIR = 'auto_generate_data'

def take_random(random_range):
	min_threshold, max_threshold = random_range
	value = np.random.randint(low=min_threshold, high=max_threshold)
	return value 

def cut_bounding_box(img_url, img_coord, output_dir):
    img = cv2.imread(img_url)
    img = cut_img_bbox(img, img_coord)
    img_name = img_url.split('/')[-1]
    img_name = os.path.join(output_dir, img_name)
    # print(img_name)
    # show_img(img)
    save_img(img, img_name)

def create_data(img_url, img_shape, output_dir):
	img_w, img_h = img_shape[0], img_shape[1]
	# print(img_w)
	source = cv2.imread(img_url)
	source_h, source_w, _ = source.shape 

	if img_w > source_w or img_h > source_h :
		# do nothing here, becaue the shape of sogoodce image is too small
		return 
	source_center_x = (source_w/2)
	source_center_y = (source_h/2)

	x1, x2 = int(source_center_x - img_w/2), int(source_center_x + img_w/2)
	y1, y2 = int(source_center_y - img_h/2), int(source_center_y + img_h/2)

	box =  (x1, y1, x2, y2)
	print(box)
	# now we have box and need to cut it with the box
	# do the final step now, good lucky
	cut_bounding_box(img_url, box, output_dir)

if __name__=='__main__':
	source = glob.glob('/home/netfpga/Documents/nhuan_cs15/car_classification/classification/cars_test/*.jpg')
	files = []
	for f in source:
		files.append(f)
	np.random.shuffle(files)


	width_range = (70, 130)
	height_range = (15, 45)


	files = files[0:1000]
	for f in files:
		width = take_random(width_range)
		height = take_random(height_range)
		img_shape = [width, height]
		print(img_shape)
		create_data(f, img_shape, OUTPUT_DIR)