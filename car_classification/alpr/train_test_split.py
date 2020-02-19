import numpy as np
import os
import glob
import random


DATA_DIR = 'data'
TEST_SUMMARY = os.path.join(DATA_DIR, 'test_summary.txt')
TRAIN_SUMMARY = os.path.join(DATA_DIR, 'train_summary.txt')
YES = 1
NO = 0


def save_dataset(data, output_url):
	with open(output_url, 'w') as wf:
		for line in data:
			wf.write(line + '\n')

if __name__=='__main__':
	train_license = []
	with open('train_test_segmented_summary/train_summary.txt', 'r') as rf:
		for line in rf:
			img = line.strip().split(' ')[0]
			train_license.append('{} {}'.format(img, YES))
	
	test_license = []
	with open('train_test_segmented_summary/test_summary.txt', 'r') as rf:
		for line in rf:
			img = line.strip().split(' ')[0]
			test_license.append('{} {}'.format(img, YES))

	no_license = glob.glob('data/no/*.jpg')
	no_license = ['{} {}'.format(x, NO) for x in no_license]
	no_license_len = len(no_license)
	no_license_train_len = int(no_license_len * 0.9)
	random.Random(12345).shuffle(no_license)
	train_no_license = no_license[0 : no_license_train_len]
	test_no_license = no_license[no_license_train_len : -1]


	train_data = train_license + train_no_license 
	test_data = test_license + test_no_license

	save_dataset(train_data, TRAIN_SUMMARY)
	save_dataset(test_data, TEST_SUMMARY)
