import numpy as np 
import tensorflow as tf 
import os
import cv2
from model import LicenseModel
from dataset import DataGenerator
import config
from utils import resize_img, show_img, draw_bbox
from time import time
import glob

import preprocess4lpdetection as detection
source_image =  'test2.jpg'

TRAIN_SUMMARY = 'original_data/train_summary.txt'
TEST_SUMMARY = 'original_data/test_summary.txt'

TRAIN_PREDICTION = 'original_data/train_prediction.txt'
TEST_PREDICTION = 'original_data/test_prediction.txt'

ALL_PREDICTION = 'original_data/all_prediction.txt'
ALL_SUMMARY = 'original_data/all_summary.txt'
NO_LICENSE_SUMMARY = 'original_data/no_license/no_license_summary.txt'
NO_LICENSE_PREDICTION = 'original_data/no_license/no_license_prediction.txt'

def reshape_img(img, w=config.INPUT_WIDTH, h=config.INPUT_HEIGHT):
	size = (w, h)
	img = resize_img(img, size)
	img = img / 255
	return np.array(img)

def read_summary(summary_file):
	result = []
	with open(summary_file, 'r') as rf:
		for line in rf:
			line = line.strip().split(',')
			img_path = line[0]
			# box = [int(x) for x in box]
			result.append(img_path)
	return result

def predict(model, detection, img_path, threshold=0.6):
	"""
		result is coord_list is shape = (number of plates, (x1, y1, x2, y2, prediction_rate))
	"""
	source_image = cv2.imread(img_path)
	img_w, img_h, c = source_image.shape
	possible_plates = detection.detect_plate(source_image)
	coord_list = []
	l = len(possible_plates)
	if l > 0:
		x_in = np.zeros((l, config.INPUT_HEIGHT, config.INPUT_WIDTH, 3))
		count = 0
		for count, plate in enumerate(possible_plates):
			img_temp = plate.get_plate_img(source_image)
			img_temp = reshape_img(img_temp)
			x_in[count] = img_temp
		pr = model.predict(x_in)
		pr = pr[..., -1]
		for i in range(len(pr)):
			if pr[i] > threshold:
				temp_box = list(possible_plates[i].get_extend_box())
				temp_box.append(pr[i])
				coord_list.append(temp_box)

	result_img = source_image

	for coord in coord_list:
		result_img = draw_bbox(result_img, coord[0:4])

	return coord_list, result_img

def create_prediction_file(model, detection, summary_file, output_file):

	data = read_summary(summary_file)
	data_len = len(data)
	with open(output_file, 'w') as wf:
		count = 0
		for img in data:
			print(img)
			coord_list, result_img = predict(model, detection, img)
			# show_img(result_img)
			coord_list = [[str(x) for x in y] for y in coord_list]
			coord_list = [' '.join(x) for x in coord_list]
			coord_list = ','.join(coord_list)
			# print(coord_list)
			if len(coord_list) > 0:
				wf.write('{},{}\n'.format(img, coord_list))
				print('{},{}\n'.format(img, coord_list))
			else:
				wf.write('{}\n'.format(img))
				print('{}\n'.format(img))
			count += 1
			# print('\n')
			if count % 50 == 0:
				print('{} at: {}/{}'.format(output_file, count, data_len))

def no_license_prediction(model, detection, output_file, source_path='original_data/no_license/'):
	# data = glob.glob(source_path + '*/*.JPEG')
	# with open(NO_LICENSE_SUMMARY, 'w') as wf:
	# 	for line in data:
	# 		wf.write('{}\n'.format(line))
	create_prediction_file(model, detection, NO_LICENSE_SUMMARY, output_file)

if __name__=='__main__':
	model = LicenseModel()
	optimizer = model.optimizer()
	model = model.model(training=False)
	model.compile(optimizer='sgd', 
              loss=tf.keras.losses.BinaryCrossentropy(),
              )
	model.load_weights('check_point/cp{}.h5'.format(5))

	# create_prediction_file(model, detection, TRAIN_SUMMARY, TRAIN_PREDICTION)
	# create_prediction_file(model, detection, TEST_SUMMARY, TEST_PREDICTION)
	coord_list, image_result = predict(model, detection, 'original_data/no_license/n02086910/n02086910_8698.JPEG', threshold=0.5)
	show_img(image_result)

	# no_license_prediction(model, detection, NO_LICENSE_PREDICTION)



