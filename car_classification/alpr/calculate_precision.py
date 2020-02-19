import numpy as np 

from predict import (
	TRAIN_SUMMARY,
	TEST_SUMMARY,
	TRAIN_PREDICTION,
	TEST_PREDICTION,
	ALL_PREDICTION,
	ALL_SUMMARY
	)

def read_summary(summary_file):
	result = []
	with open(summary_file, 'r') as rf:
		for line in rf:
			result.append(line.strip())
	return result

def compute_iou(predict_box, ground_true_box):
	x1_t, y1_t, x2_t, y2_t = ground_true_box
	x1_p, y1_p, x2_p, y2_p = predict_box

	if (x1_p > x2_p) or (y1_p > y2_p):
		raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
	if (x1_t > x2_t) or (y1_t > y2_t):
		raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))
	if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
		return 0.0

	far_x = np.min([x2_t, x2_p])
	near_x = np.max([x1_t, x1_p])
	far_y = np.min([y2_t, y2_p])
	near_y = np.max([y1_t, y1_p])

	inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
	true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
	pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
	iou = inter_area / (true_box_area)# + pred_box_area - inter_area)

	return iou

def ground_true_search(ground_true_summary_file):
	data = read_summary(ground_true_summary_file)
	ground_true_dictionary = {}
	for line in data:
		line = line.split(',')
		img_path = line[0].strip()
		box = line[1].split()
		box = [int(x) for x in box]
		ground_true_dictionary[img_path] = box
	return ground_true_dictionary

def calculate_a_image(data, ground_true_search, iou_threshold=0.7):
	data = data.split(',')
	img_path = data[0].strip()
	ground_true_box = ground_true_search[img_path]
	special_number = 0
	if len(data) == 1 :
		true_pos = 0
		false_pos = 1
		false_neg = 0
		return {'true_pos': true_pos, 'false_pos': false_pos, 'false_neg': false_neg, 'special_number': special_number}

	pred_boxes = data[1:]
	true_pos, false_pos, false_neg = 0, 0, 0
	for pred_box in pred_boxes:
		pred_box = (pred_box.strip().split())[0:4]
		pred_box = [int(x) for x in pred_box]
		iou = compute_iou(pred_box, ground_true_box)
		# print(pred_box, ground_true_box, iou)
		if iou > iou_threshold:
			true_pos += 1
			false_pos += 0
			false_neg += 0
			special_number += 1
		else :
			true_pos += 0
			false_pos += 1
			false_neg += 0
			# print(img_path)
	return {'true_pos': true_pos, 'false_pos': false_pos, 'false_neg': false_neg, 'special_number': special_number}

def calculate_precision_recall(summary, prediction):
	searcher = ground_true_search(summary)
	# print(searcher)
	predict_data = read_summary(prediction)
	true_pos, false_pos, false_neg = 0, 0, 0
	special_number = 0
	for data in predict_data:
		result = calculate_a_image(data, searcher)
		true_pos += result['true_pos']
		false_pos += result['false_pos']
		false_neg += result['false_neg']
		special_number += result['special_number']
	precision = true_pos*1.0 / (true_pos + false_pos + 0.0000000000000001)
	recall = true_pos*1.0 / (true_pos + false_neg + 0.0000000000000001)
	haha = (special_number/len(predict_data))
	return precision, recall, haha

if __name__=='__main__':
	# print(calculate_precision_recall(TRAIN_SUMMARY, TRAIN_PREDICTION))
	# print(calculate_precision_recall(TEST_SUMMARY, TEST_PREDICTION))
	print(calculate_precision_recall(ALL_SUMMARY, ALL_PREDICTION))