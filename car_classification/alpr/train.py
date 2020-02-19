import numpy as np 
import tensorflow as tf 
import os

from model import LicenseModel
from dataset import DataGenerator
from train_test_split import TEST_SUMMARY, TRAIN_SUMMARY

LOG_PATH = "check_point/log.txt"
def write_log(data, log_path):
    with open(log_path, "a") as myfile:
        data = [str(x) for x in data]
        myfile.write('\t'.join(data) + '\n')

if __name__=='__main__':
	model = LicenseModel()
	optimizer = model.optimizer()
	model = model.model(training=True)
	model.compile(optimizer=optimizer, 
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"]
              )
	# model.load_weights('check_point/cp{}.h5'.format(50))
	train_data = DataGenerator(TRAIN_SUMMARY)
	steps_per_epoch = train_data.number_of_training_batches
	
	test_data = DataGenerator(TEST_SUMMARY)
	with open(LOG_PATH, 'w') as wf:
		wf.write('{}\t{}\t{}\t{}\t{}\n'.format('step', 'train_loss', 'train_acc', 'test_loss', 'test_acc'))
	for i in range(50):
		train = train_data.data_generator(training=True)
		test = test_data.data_generator(training=False)
		history = model.fit_generator(train, steps_per_epoch=steps_per_epoch, epochs=1)

		train_loss = (history.history['loss'])[0]
		train_acc = (history.history['accuracy'])[0]
		test_loss = []
		test_acc = []

		for x_test, y_test in test:
			loss, acc = model.evaluate(x_test, y_test, verbose=False)
			test_loss.append(loss)
			test_acc.append(acc)

		test_loss = np.mean(test_loss)
		test_acc = np.mean(test_acc)
		print("step {}\ttrain loss {}\ttrain acc {}\ttest loss {}\ttest acc {}".format(i + 1, train_loss, train_acc, test_loss, test_acc))
		write_log([i, train_loss, train_acc, test_loss, test_acc], LOG_PATH)
		print('step{}'.format(i + 1))
		# print(test_results)
		if ((i+1) % 5) == 0 :
			checkpoint_path = "check_point/cp{}.h5".format(i + 1)
			model.save_weights(checkpoint_path)
