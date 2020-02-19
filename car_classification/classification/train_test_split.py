import os
import numpy as np 
import glob
import cv2
from utils import show_img
import random


TRAIN_SUMMARY_PATH = 'data/train_summary.txt'
TEST_SUMMARY_PATH = 'data/test_summary.txt'
IS_CAR = 1
IS_NOT_CAR = 0

if __name__=='__main__':
	car_train = list(glob.glob('data/train/car/*/*.jpg'))
	not_car_train = list(glob.glob('data/train/not_car/*/*.JPEG'))
	car_test = list(glob.glob('data/test/car/*/*.jpg'))
	not_car_test = list(glob.glob('data/test/not_car/*/*.JPEG'))

	# not_car = not_car_train + not_car_test
	# random.Random(123).shuffle(not_car)
	# split = int(len(not_car) / 2)
	# not_car_train = not_car[0: split]
	# not_car_test = not_car[split:]

	# print(not_car)
	with open(TRAIN_SUMMARY_PATH, 'w') as wf:
		for car in car_train:
			wf.write(car + ' {}\n'.format(IS_CAR))
		for car in not_car_train:
			wf.write(car + ' {}\n'.format(IS_NOT_CAR))



	
	with open(TEST_SUMMARY_PATH, 'w') as wf:
		for car in car_test:
			wf.write(car + ' {}\n'.format(IS_CAR))
		for car in not_car_test:
			wf.write(car + ' {}\n'.format(IS_NOT_CAR))
	