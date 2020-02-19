import numpy as np 
import tensorflow as tf 
import glob
import cv2
from model import TiniYOLO
from dataset import DataGenerator, transform_img, decode_bounding_box
from loss import tini_yolo_loss
from config import GRID_SIZE
from utils import draw_bbox, show_img

def decode_output(y_pred, shape, grid_size):
	y_pred[..., 0:2] = tf.sigmoid(y_pred[..., 0:2])
	y_pred[..., 4] = tf.sigmoid(y_pred[..., 4])
	box = decode_bounding_box(y_pred[0], shape, grid_size)
	return box


def load_data(data_folder='./vietnam/'):
	files = glob.glob(data_folder + '/*.jpg')
	return files

if __name__=='__main__':
	files = load_data()
	print(files)


	model = TiniYOLO()
	model = model.model(training=False)
	model.load_weights('big_img_check_point/cp{}.h5'.format(400))

	for sample in files:
		sample = cv2.imread(sample)
		sample = transform_img(sample)
		sample_shape = sample.shape
		result = np.expand_dims(sample, 0)
		
		result = model.predict(result)
		result = decode_output(result, sample_shape, GRID_SIZE)
		img = draw_bbox(sample, result)
		show_img(img, str(sample_shape))
		print(result)