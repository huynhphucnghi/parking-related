import glob
import pickle
import cv2
import numpy as np
from time import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from dask import delayed


INPUT_WIDTH = 50
INPUT_HEIGHT = 15


def read_dataset():
	license = glob.glob('./data/yes/*.jpg')
	no_license = glob.glob('./data/no/*.jpg')

	X, Y = [], []
	for haha in license:
		X.append(haha)
		Y.append(1)

	for haha in no_license:
		X.append(haha)
		Y.append(0)
	return X, Y

def show_img(img, title='Fuck You, Draw more'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_bbox(img, box, color=(255, 0, 0)):
    box = [int(x) for x in box]
    return cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color)

@delayed
def extract_feature_image(img, feature_types, feature_coords):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_types,
                             feature_coord=feature_coords)
def resize_img(img, size=(INPUT_WIDTH, INPUT_HEIGHT)):
    img_w, img_h = size
    img = cv2.resize(img, (img_w, img_h))
    return img

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def haar_feature(img_path, feature_types, feature_coords):
	img = cv2.imread(img_path)
	img = rgb2gray(img)
	img = resize_img(img)
	
	return extract_feature_image(img, feature_types, feature_coords)

def get_haar_coord():
	feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']

	# Extract all possible features
	list_feature_coords, list_feature_types = \
	    haar_like_feature_coord(width=INPUT_WIDTH, height=INPUT_HEIGHT,
                            feature_type=feature_types)
	return list_feature_coords, list_feature_types

def save_data():
	X, Y = read_dataset()
	print(np.shape(X), np.shape(Y))
	feature_coords, feature_types = get_haar_coord()
	# X = X[0:10]
	start = time()
	X = delayed(haar_feature(img, feature_types, feature_coords) for img in X)
	X = np.array(X.compute(scheduler='threads'))
	end = time()

	print('compute_time: {}'.format(end - start))
	
	data = (X, Y)
	with open('extracted_features_data.pickle', 'wb') as wf:
		pickle.dump(data, wf, protocol=pickle.HIGHEST_PROTOCOL)

def load_data():
	with open('extracted_features_data.pickle', 'rb') as rf:
		X, Y = pickle.load(rf)
		print('X shape ', np.shape(X), '  Y shape',np.shape(Y))
	return X, Y

def save_model(model, step):
	with open('adaboost_classifier_{}.pickle'.format(step), 'wb') as wf:
		pickle.dump(model, wf, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(step):
	with open('adaboost_classifier_{}.pickle'.format(step), 'rb') as rf:
		return pickle.load(rf)

def take_important_features(data, index):
	new_data = []
	for d in data:
		new_data.append(d[index])
	return new_data

def save_reduced_data(X, Y, index):
	data = (X, Y, index)
	with open('adaboost_reduced_features_data.pickle', 'wb') as wf:
		pickle.dump(data, wf, protocol=pickle.HIGHEST_PROTOCOL)

def load_reduced_data():
	with open('adaboost_reduced_features_data.pickle', 'rb') as rf:
		X, Y, index = pickle.load(rf)
		print('X shape ', np.shape(X), '  Y shape',np.shape(Y))
	return X, Y, index

def prepare_feature(img_path, index):
	feature_coords, feature_types = get_haar_coord()
	feature_coords = feature_coords[index]
	feature_types = feature_types[index]
	
	X = delayed(haar_feature(img_path, feature_types, feature_coords))
	X = np.array(X.compute(scheduler='threads'))
	
	return X

if __name__ == '__main__':
	# just run one if you want to extract feature from data
	# save_data()

	X, Y, idx = load_reduced_data()
	# # # X = X[..., 0:10]
	# x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=500,
 #                                                    random_state=0)
	# model = AdaBoostClassifier(n_estimators=1000, random_state=0)

	# model.fit(x_train, y_train)

	# # using to run one when training, after that just load model from file
	# save_model(model, 'reduced_feat')
	X, Y = read_dataset()
	model = load_model('reduced_feat')
	
	

	start = time()
	feat = prepare_feature(X[0], idx)
	y = model.predict([feat])
	end = time()
	print(end - start, y)
	# y_train_pred = model.predict(x_train)
	# y_test_pred = model.predict(x_test)
	# print('train accuracy : ', accuracy_score(y_train_pred, y_train))
	# print('test  accuracy : ', accuracy_score(y_test_pred, y_test))
	# idx_sorted = np.argsort(model.feature_importances_)[::-1]
	# idx_sorted = idx_sorted[0:700]
	# X = take_important_features(X, idx_sorted)

	# save_reduced_data(X, Y, idx_sorted)
	# count = 0
	# for i in model.feature_importances_:
	#     if i > 0.0 :
	#         count += 1
	# print(count)

	# count = 0
	# for i in model.estimator_weights_ :
	#     if i > 0.0 :
	#         count += 1
	# print(count)