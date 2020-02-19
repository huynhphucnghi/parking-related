import numpy as np 
import tensorflow as tf 
import os
import cv2
from model_01.model import TiniYOLO
from dataset import DataGenerator, transform_img, decode_bounding_box, read_summary
from loss import tini_yolo_loss
from config import GRID_SIZE, IMAGE_SIZE
from utils import draw_bbox, show_img
from train_test_split import TEST_DATA_SUMMARY, TRAIN_DATA_SUMMARY
# EXAMPLE_IMAGE = './data/us/wts-lg-000030.jpg'
MODEL_CHECK_POINT_PATH = 'model_01/check_point/'
OUTPUT_PATH = 'model_01_result/'

def decode_output(y_pred, shape, grid_size):
	y_pred[..., 0:2] = tf.sigmoid(y_pred[..., 0:2])
	y_pred[..., 4] = tf.sigmoid(y_pred[..., 4])
	box = decode_bounding_box(y_pred[0], shape, grid_size)
	return box

def write_result(data, file_path):
	with open(file_path, 'a') as af:
		af.write('{}\n'.format(data))

def get_original_bbox(box, original_shape, image_size):
	box = list(box)
	h, w, _ = original_shape
	h_ratio = h / image_size
	w_ratio = w / image_size
	box[0] = box[0] * w_ratio
	box[2] = box[2] * w_ratio
	box[1] = box[1] * h_ratio
	box[3] = box[3] * h_ratio
	
	return [int(x) for x in box]

if __name__=='__main__':
	input_path = os.path.join(OUTPUT_PATH, 'test_summary_yes.txt')
	output_path = os.path.join(OUTPUT_PATH, 'test_summary_yes_predict.txt')
	data = read_summary(input_path)
	data = [line.strip().split(',')[0] for line in data]
	print(data[0:10])
	model = TiniYOLO()
	model = model.model(training=False)
	model.load_weights( MODEL_CHECK_POINT_PATH + 'cp{}.ckpt'.format(75))

	for file_path in data:
		# bbox = bbox.split(' ')
		# bbox = [int(x) for x in bbox]
		sample = source = cv2.imread(file_path)
		sample = transform_img(sample, IMAGE_SIZE)
		sample_shape = sample.shape
		result = np.expand_dims(sample, 0)
		
		result = model.predict(result)
		result = decode_output(result, sample_shape, GRID_SIZE)

		source_shape = source.shape
		result = get_original_bbox(result, source_shape, IMAGE_SIZE)

		img = draw_bbox(source, result)
		# show_img(img, str(source_shape))
		print(result)
		if (result[0] == 0) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) :
			write_result(file_path, output_path)
			pass
		else :
			result = [str(x) for x in result]
			haha = '{},{}'.format(file_path, ' '.join(result))
			write_result(haha, output_path)
		# break
