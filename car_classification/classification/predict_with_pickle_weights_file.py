import numpy as np 
import tensorflow as tf 
import os
import pickle
from model_04.model import TiniVgg
from dataset import standardize_img
from utils import show_img, set_weights
from config import INPUT_SIZE 
from dataset import DataGenerator, TEST_SUMMARY_PATH

PICKLE_WEIGHT_FILE = './pickle_weight/weights.pickle'
def set_model_weight(model, pickle_weight_file):
	with open(pickle_weight_file, 'rb') as rf:
		weights_dict = pickle.load(rf)
	for key in weights_dict.keys():
		set_weights(model, key, weights_dict[key])
	return model

def prepare_image(image_path):
	return np.expand_dims(standardize_img(image_path, INPUT_SIZE, INPUT_SIZE), 0)

if __name__=='__main__':
	model = TiniVgg()
	model = model.model(training=False)
	model = set_model_weight(model, PICKLE_WEIGHT_FILE)

	# load dataset
	dataset = DataGenerator(TEST_SUMMARY_PATH, batch_size=1)
	data = dataset.alldata
	total_sample = len(data)
	correct_number_iscar = 0
	correct_number_isnotcar = 0
	# print(total_sample)
	print(data[0])
	with open('predict_result.txt', 'w') as rsf:
		for i, (x, y) in enumerate(data) :
			if (i+1)%100 == 0:
				print(i)
			y = int(y) # cast to integer
			input_image = prepare_image(x)
			output_predict = model.predict(input_image)
			output_predict = output_predict[0][0]
			if output_predict > 0.5:
				label_predict = 1
			else :
				label_predict = 0
			if label_predict == y and y == 1 :
				correct_number_iscar += 1
			if label_predict == y and y == 0:
				correct_number_isnotcar += 1
			# rsf.write(x + '\t  {} \t {} \t {} \n'.format(y, label_predict, output_predict) ) 
		    # show_img(input_image[0], str(output_predict))
		print('total samples is {}, with correct car is {} and correct notcar is {} \n'.format(total_sample, correct_number_iscar, correct_number_isnotcar))
		rsf.write('total samples is {}, with correct car is {} and correct notcar is {} \n'.format(total_sample, correct_number_iscar, correct_number_isnotcar))