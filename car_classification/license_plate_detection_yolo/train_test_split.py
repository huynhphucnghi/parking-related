
import numpy as np
import random
import glob

TRAIN_DATA_SUMMARY = './summary/train_summary.txt'
TEST_DATA_SUMMARY = './summary/test_summary.txt'


def save_file(data, file_path):
	with open(file_path, 'w') as wf:
		for line in data:
			wf.write(line + '\n')

if __name__ == "__main__":

	license_train = []
	with open('./data/license_train_summary.txt', 'r') as rf:
		for line in rf:
			license_train.append(line.strip())

	license_test = []
	with open('./data/license_test_summary.txt', 'r') as rf:
		for line in rf:
			license_test.append(line.strip())

	no_license_train = glob.glob('data/no_license/train/*/*.JPEG')

	no_license_test = glob.glob('data/no_license/test/*/*.JPEG')
	# print(len(license_train), len(license_test), len(no_license_train), len(no_license_test))
	train_data = license_train + no_license_train
	test_data = license_test + no_license_test

	save_file(train_data, TRAIN_DATA_SUMMARY)
	save_file(test_data, TEST_DATA_SUMMARY)