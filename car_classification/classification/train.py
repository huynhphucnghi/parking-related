import numpy as np 
import tensorflow as tf 
import os

from model_04.model import TiniVgg
from dataset import DataGenerator, TRAIN_SUMMARY_PATH, TEST_SUMMARY_PATH

MODEL_CHECK_POINT_PATH = 'model_04/check_point/'
LOG_PATH = MODEL_CHECK_POINT_PATH + "log.txt"
def write_log(data, log_path):
    with open(log_path, "a") as myfile:
        data = [str(x) for x in data]
        myfile.write('\t'.join(data) + '\n')

if __name__=='__main__':
    model = TiniVgg()
    optimizer = model.optimizer()
    model = model.model(training=True)
    model.summary()
    model.compile(optimizer=optimizer, 
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])
    # model.load_weights( MODEL_CHECK_POINT_PATH + 'cp{}.ckpt'.format(50))
    train_dataset = DataGenerator(TRAIN_SUMMARY_PATH)
    test_dataset = DataGenerator(TEST_SUMMARY_PATH)

    steps_per_epoch = train_dataset.number_of_batches
    with open(LOG_PATH, 'w') as wf:
        wf.write('{}\t{}\t{}\t{}\t{}\n'.format('step', 'train_loss', 'train_acc', 'test_loss', 'test_acc'))
    for i in range(20, 50):
        train_data = train_dataset.data_generator(training=True, img_augmentation=True)
        test_data = test_dataset.data_generator(training=False)
        model.optimizer.lr = 0.0001
        history = model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, epochs=1)
        train_loss = (history.history['loss'])[0]
        train_acc = (history.history['accuracy'])[0]
        test_loss = []
        test_acc = []
        for x_test, y_test in test_data:
            loss, acc = model.evaluate(x_test, y_test, verbose=False)
            test_loss.append(loss)
            test_acc.append(acc)
            # break
        test_loss = np.mean(test_loss)
        test_acc = np.mean(test_acc)
        print("step {}\ttrain loss {}\ttrain acc {}\ttest loss {}\ttest acc {}".format(i + 1, train_loss, train_acc, test_loss, test_acc))
        write_log([i, train_loss, train_acc, test_loss, test_acc], LOG_PATH)
        if ((i+1) % 5) == 0 :
            checkpoint_path = MODEL_CHECK_POINT_PATH + "cp{}.ckpt".format(i + 1)
            model.save_weights(checkpoint_path)