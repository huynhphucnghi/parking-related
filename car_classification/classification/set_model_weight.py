import pickle
import numpy as np
from model import TiniVgg
from utils import set_weights

PICKLE_WEIGHT_FILE = './pickle_weight/weights.pickle'

def set_model_weight(model, pickle_weight_file):
	with open(pickle_weight_file, 'rb') as rf:
		weights_dict = pickle.load(rf)
	for key in weights_dict.keys():
		set_weights(model, key, weights_dict[key])
	return model

if __name__ == '__main__':
	model = TiniVgg()
	model = model.model(training=False)
	model = set_model_weight(model, PICKLE_WEIGHT_FILE)
	