from model_04.model import TiniVgg
from utils import get_conv_layer, get_bn_layer, get_dense_layer
import pickle


MODEL_CHECK_POINT_PATH = 'model_04/check_point/'

if __name__=='__main__':
	model = TiniVgg()
	model = model.model(training=False)
	model.load_weights(MODEL_CHECK_POINT_PATH + "cp{}.ckpt".format(30))
	weight_dict = {}
	model_layer_name = []
	for layer in model.layers:
		model_layer_name.append(layer.name)

	for i in range(len(model_layer_name)):
		if 'conv' in model_layer_name[i]:
			conv_weights = get_conv_layer(model, model_layer_name[i])
			weight_dict[model_layer_name[i]] = conv_weights
		if 'batchnorm' in model_layer_name[i]:
			bn_weights = get_bn_layer(model, model_layer_name[i])
			weight_dict[model_layer_name[i]] = bn_weights
		if 'dense' in model_layer_name[i]:
			dense_weights = get_dense_layer(model, model_layer_name[i])
			weight_dict[model_layer_name[i]] = dense_weights
	with open('./pickle_weight/weights.pickle', 'wb') as wf:
		pickle.dump(weight_dict, wf, protocol=pickle.HIGHEST_PROTOCOL)